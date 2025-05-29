from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pickle
import torch
import pandas as pd
from sklearn.model_selection import GroupKFold
from transformers import AutoProcessor, AutoModelForImageClassification
from torch.nn.functional import softmax
from data_loader import allen_api
import os
from dotenv import load_dotenv
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from skbio.stats.composition import clr
from joblib import Parallel, delayed

load_dotenv()

class PipelineStep(ABC):
    @abstractmethod
    def process(self, data):
        pass

class AllenStimuliFetchStep(PipelineStep):
    SESSION_A = 501704220
    SESSION_B = 501559087
    SESSION_C = 501474098

    def __init__(self, boc):
        self.boc = boc

    def process(self, data):
        if isinstance(data, tuple):
            container_id, session, stimulus = data
            data = {'container_id': container_id, 'session': session, 'stimulus': stimulus}
        elif data is None:
            data = {}

        raw_data_dct = {
            'natural_movie_one': self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template('natural_movie_one'),
            'natural_movie_two': self.boc.get_ophys_experiment_data(self.SESSION_C).get_stimulus_template('natural_movie_two'),
            'natural_movie_three': self.boc.get_ophys_experiment_data(self.SESSION_A).get_stimulus_template('natural_movie_three'),
            'natural_scenes': self.boc.get_ophys_experiment_data(self.SESSION_B).get_stimulus_template('natural_scenes')
        }

        data['raw_data_dct'] = raw_data_dct
        return data

class AllenNeuralResponseExtractor(PipelineStep):
    def __init__(self, boc, eid_dict, stimulus_session_dict, threshold=0.0):
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.threshold = threshold

    def process(self, data):
        container_id = data['container_id']
        session = data['session']
        stimulus = data['stimulus']

        valid_stims = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stims:
            raise ValueError(f"Stimulus '{stimulus}' not valid for session '{session}'. Valid: {valid_stims}")

        session_eid = self.eid_dict[container_id][session]
        dataset = self.boc.get_ophys_experiment_data(session_eid)
        dff_traces = self.boc.get_ophys_experiment_events(ophys_experiment_id=session_eid)
        stim_table = dataset.get_stimulus_table(stimulus)

        X_list, frame_list = [], []

        for _, row in stim_table.iterrows():
            if row['frame'] == -1:
                continue
            start_t, end_t = row['start'], row['end']
            frame_idx = row['frame']
            time_indices = range(start_t, end_t)

            if len(time_indices) == 0:
                trial_vector = np.zeros(dff_traces.shape[0])
            else:
                relevant_traces = dff_traces[:, time_indices]
                trial_vector = np.max(relevant_traces, axis=1)
                trial_vector = (trial_vector > self.threshold).astype(float)

            X_list.append(trial_vector)
            frame_list.append(frame_idx)

        data['X_neural'] = np.vstack(X_list)
        data['frame_ids'] = np.array(frame_list)
        return data

class ImageToEmbeddingStep(PipelineStep):
    def __init__(self, embedding_cache_dir: str):
        self.model_name = "google/vit-base-patch16-224"
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
        self.model.eval()

        self.embedding_cache_dir = Path(embedding_cache_dir)
        self.embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        self.model_prefix = self.model_name.replace('/', '_')
        self.embeddings_file = self.embedding_cache_dir / f"{self.model_prefix}_embeddings_softmax.pkl"

    def process(self, data):
        raw_data_dct = data['raw_data_dct']

        if self.embeddings_file.exists():
            print(f"Found existing embeddings for model {self.model_prefix}. Using file:\n {self.embeddings_file}")
            data['embedding_file'] = str(self.embeddings_file)
            return data

        print(f"No cache found for model {self.model_prefix}. Computing now...")
        embeddings_dict = {}
        for stim_name, frames_array in raw_data_dct.items():
            embeddings = self._process_stims(frames_array)
            embeddings_dict[stim_name] = embeddings

        with open(self.embeddings_file, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        print(f"Saved embeddings to {self.embeddings_file}")

        data['embedding_file'] = str(self.embeddings_file)
        return data

    def _process_stims(self, frames_array):
        n_frames = len(frames_array)
        frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)

        with torch.no_grad():
            inputs = self.processor(images=frames_3ch[0], return_tensors="pt")
            outputs = self.model(**inputs)
            n_classes = outputs.logits.shape[-1]

        all_probs = np.empty((n_frames, n_classes), dtype=np.float32)
        for i in range(n_frames):
            inputs = self.processor(images=frames_3ch[i], return_tensors="pt")
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = softmax(logits, dim=-1).squeeze().cpu().numpy()
            all_probs[i, :] = probs

        return all_probs

class AllenViTRegressionDatasetBuilder(PipelineStep):
    def __init__(self):
        pass

    def process(self, data):
        embedding_file = data['embedding_file']
        stimulus = data['stimulus']
        frame_ids = data['frame_ids']

        with open(embedding_file, 'rb') as f:
            all_stim_embeddings = pickle.load(f)

        embed_array = all_stim_embeddings[stimulus]
        X_embed = np.array([embed_array[f_idx] for f_idx in frame_ids], dtype=np.float32)

        data['X_embed'] = X_embed
        return data

class CompressionBottleneckAnalysisStep(PipelineStep):
    def __init__(self, Cs=np.logspace(-3, 1, 10), tuning_sample_fraction=0.5, n_jobs=-1):
        self.Cs = Cs
        self.tuning_sample_fraction = tuning_sample_fraction
        self.n_jobs = n_jobs

    def process(self, data):
        X_embed_raw = data['X_embed']
        X_neural = data['X_neural']
        X_embed_real = clr(X_embed_raw + 1e-6)
        X_embed_perm = np.random.permutation(X_embed_real)

        # Filter neurons with only one class
        valid_indices = [i for i in range(X_neural.shape[1]) if len(np.unique(X_neural[:, i])) > 1]
        X_neural = X_neural[:, valid_indices]

        # Subsample for tuning
        n_samples = X_embed_real.shape[0]
        sample_size = int(self.tuning_sample_fraction * n_samples)
        rng = np.random.default_rng(seed=123)
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)

        Xr, Xp = X_embed_real[sample_indices], X_embed_perm[sample_indices]
        Yr = X_neural[sample_indices]

        def fit_ll(X, y, C):
            if len(np.unique(y)) < 2:
                return float('-inf')
            try:
                clf = LogisticRegression(penalty='l2', C=C, solver='saga', max_iter=1000)
                clf.fit(X, y)
                prob = clf.predict_proba(X)
                ll = -log_loss(y, prob, labels=[0, 1], normalize=True)
                return ll
            except Exception:
                return float('-inf')

        def score_C(C):
            real_lls = []
            perm_lls = []
            for i in range(Yr.shape[1]):
                y = Yr[:, i]
                ll_real = fit_ll(Xr, y, C)
                ll_perm = fit_ll(Xp, y, C)
                real_lls.append(ll_real)
                perm_lls.append(ll_perm)
            real_lls = np.array(real_lls)
            perm_lls = np.array(perm_lls)
            delta = real_lls - perm_lls
            return {
                'C': C,
                'real': real_lls.tolist(),
                'perm': perm_lls.tolist(),
                'delta': delta.tolist(),
                'mean_delta': np.mean(delta)
            }

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(score_C)(C) for C in self.Cs
        )

        best_result = max(results, key=lambda x: x['mean_delta'])

        # Store all results
        data['compression_bottleneck_results'] = results
        data['best_C_bottleneck'] = best_result['C']
        data['max_mean_delta'] = best_result['mean_delta']

        print(f"Best C (bottleneck): {best_result['C']}")
        print(f"Max mean delta (log-likelihood real - permuted): {best_result['mean_delta']:.5f}")

        # Inside CompressionBottleneckAnalysisStep.process() before return data
        save_path = Path("compression_bottleneck_results.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({
                'all_results': results,
                'best_C': best_result['C'],
                'max_mean_delta': best_result['mean_delta']
            }, f)

        print(f"Saved compression bottleneck results to {save_path.resolve()}")
        return data

class AnalysisPipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, data):
        for step in self.steps:
            data = step.process(data)
        return data

def make_container_dict(boc):
    experiment_container = boc.get_experiment_containers()
    container_ids = [dct['id'] for dct in experiment_container]
    eids = boc.get_ophys_experiments(experiment_container_ids=container_ids)
    df = pd.DataFrame(eids)
    reduced_df = df[['id', 'experiment_container_id', 'session_type']]
    grouped_df = reduced_df.groupby(['experiment_container_id', 'session_type'])['id'].agg(list).reset_index()
    eid_dict = {}
    for row in grouped_df.itertuples(index=False):
        c_id, sess_type, ids = row
        if c_id not in eid_dict:
            eid_dict[c_id] = {}
        eid_dict[c_id][sess_type] = ids[0]
    return eid_dict

if __name__ == '__main__':
    boc = allen_api.get_boc()
    eid_dict = make_container_dict(boc)
    stimulus_session_dict = {
        'three_session_A': ['natural_movie_one', 'natural_movie_three'],
        'three_session_B': ['natural_movie_one', 'natural_scenes'],
        'three_session_C': ['natural_movie_one', 'natural_movie_two'],
        'three_session_C2': ['natural_movie_one', 'natural_movie_two']
    }

    embedding_cache_dir = os.environ.get('TRANSF_EMBEDDING_PATH', 'embeddings_cache')
    container_id = list(eid_dict.keys())[0]
    session = list(eid_dict[container_id].keys())[0]
    stimulus = stimulus_session_dict.get(session, [])[0]
    session='three_session_B'
    stimulus='natural_scenes'
    print(f"Running pipeline for container_id={container_id}, session={session}, stimulus={stimulus}")

    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),
        ImageToEmbeddingStep(embedding_cache_dir),
        AllenNeuralResponseExtractor(boc, eid_dict, stimulus_session_dict),
        AllenViTRegressionDatasetBuilder(),
        CompressionBottleneckAnalysisStep()
    ])
    import time
    start=time.time()
    result = pipeline.run((container_id, session, stimulus))
    end=time.time()
    print('Time taken', end-start)