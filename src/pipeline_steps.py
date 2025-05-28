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

class CSelectionTunerStep(PipelineStep):
    def __init__(self, Cs=np.logspace(-3, 1, 6), num_tune_neurons=10, tuning_sample_fraction=0.5, n_jobs=-1):
        self.Cs = Cs
        self.num_tune_neurons = num_tune_neurons
        self.tuning_sample_fraction = tuning_sample_fraction
        self.n_jobs = n_jobs

    def process(self, data):
        X_embed_raw = data['X_embed']
        X_embed = clr(X_embed_raw + 1e-6)
        X_neural = data['X_neural']

        # Filter out neurons with only one class
        valid_indices = [i for i in range(X_neural.shape[1]) if len(np.unique(X_neural[:, i])) > 1]
        X_neural = X_neural[:, valid_indices]

        n_neurons = X_neural.shape[1]
        neuron_indices = np.arange(n_neurons)
        rng = np.random.default_rng(seed=42)
        tune_indices = rng.choice(neuron_indices, size=min(self.num_tune_neurons, n_neurons), replace=False)

        # Subsample data for tuning
        n_samples = X_embed.shape[0]
        sample_size = int(self.tuning_sample_fraction * n_samples)
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
        X_embed_tune = X_embed[sample_indices]
        X_neural_tune = X_neural[sample_indices]

        def fit_and_score(i, C, penalty):
            y = X_neural_tune[:, i]
            if len(np.unique(y)) < 2:
                return None
            try:
                clf = LogisticRegression(penalty=penalty, C=C, solver='saga', max_iter=1000)
                clf.fit(X_embed_tune, y)
                prob = clf.predict_proba(X_embed_tune)
                ll = -log_loss(y, prob, labels=[0, 1], normalize=True)
                return (C, ll)
            except Exception:
                return None

        best_Cs = {}
        for penalty in ['l1', 'l2']:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(fit_and_score)(i, C, penalty)
                for i in tune_indices
                for C in self.Cs
            )

            C_scores = {C: [] for C in self.Cs}
            for result in results:
                if result is not None:
                    C, ll = result
                    C_scores[C].append(ll)

            avg_scores = [(np.mean(vals), C) for C, vals in C_scores.items() if vals]
            best_C = max(avg_scores, key=lambda x: x[0])[1] if avg_scores else self.Cs[0]
            best_Cs[penalty] = best_C

        data['best_Cs'] = best_Cs
        print(f"Selected C for L1: {best_Cs['l1']}")
        print(f"Selected C for L2: {best_Cs['l2']}")
        return data

class L1vRidgeGLMFitStep(PipelineStep):
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def process(self, data):
        X_embed_raw = data['X_embed']
        X_embed = clr(X_embed_raw + 1e-6)
        X_neural = data['X_neural']
        best_Cs = data['best_Cs']

        # Filter again
        valid_indices = [i for i in range(X_neural.shape[1]) if len(np.unique(X_neural[:, i])) > 1]
        X_neural = X_neural[:, valid_indices]

        def fit_model_for_neuron(i):
            y = X_neural[:, i]
            if len(np.unique(y)) < 2:
                return {'l1': float('-inf'), 'l2': float('-inf')}
            scores = {}
            for penalty in ['l1', 'l2']:
                try:
                    clf = LogisticRegression(penalty=penalty, C=best_Cs[penalty], solver='saga', max_iter=1000)
                    clf.fit(X_embed, y)
                    prob = clf.predict_proba(X_embed)
                    ll = -log_loss(y, prob, labels=[0, 1], normalize=True)
                    scores[penalty] = ll
                except Exception:
                    scores[penalty] = float('-inf')
            return scores

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_model_for_neuron)(i) for i in range(X_neural.shape[1])
        )

        avg_log_likelihoods = {'l1': [], 'l2': []}
        for res in results:
            avg_log_likelihoods['l1'].append(res['l1'])
            avg_log_likelihoods['l2'].append(res['l2'])

        data['avg_log_likelihoods'] = avg_log_likelihoods

        with open(Path("likelihoods_summary_natural_scenes.pkl"), 'wb') as f:
            pickle.dump(avg_log_likelihoods, f)
        print("Saved average log-likelihoods to likelihoods_summary.pkl")
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
        CSelectionTunerStep(),
        L1vRidgeGLMFitStep(n_jobs=-1)
    ])
    import time
    start=time.time()
    result = pipeline.run((container_id, session, stimulus))
    end=time.time()
    print("Embed shape:", result['X_embed'].shape)
    print("Neural shape:", result['X_neural'].shape)
    print("L1 log-likelihoods:", result['avg_log_likelihoods']['l1'])
    print("Ridge log-likelihoods:", result['avg_log_likelihoods']['l2'])
    print('Time taken', end-start)
'''
    
class L1vRidgeGLMFitStep(PipelineStep):
    def __init__(self, Cs=np.logspace(-3, 1, 6), num_tune_neurons=10, num_models_per_neuron=10, tuning_sample_fraction=0.5):
        self.Cs = Cs
        self.num_tune_neurons = num_tune_neurons
        self.num_models_per_neuron = num_models_per_neuron
        self.tuning_sample_fraction = tuning_sample_fraction

    def process(self, data):
        X_embed_raw = data['X_embed']
        X_embed = clr(X_embed_raw + 1e-6)  # Add small value to avoid log(0)
        X_neural = data['X_neural']
        n_neurons = X_neural.shape[1]

        neuron_indices = np.arange(n_neurons)
        rng = np.random.default_rng(seed=42)
        tune_indices = rng.choice(neuron_indices, size=min(self.num_tune_neurons, n_neurons), replace=False)

        # Optionally subsample the data for tuning
        n_samples = X_embed.shape[0]
        sample_size = int(self.tuning_sample_fraction * n_samples)
        sample_indices = rng.choice(n_samples, size=sample_size, replace=False)
        X_embed_tune = X_embed[sample_indices]
        X_neural_tune = X_neural[sample_indices]

        # Tune C separately for L1 and L2
        best_Cs = {'l1': None, 'l2': None}
        for penalty in ['l1', 'l2']:
            losses = []
            for i in tune_indices:
                y = X_neural_tune[:, i]
                clf = GridSearchCV(
                    LogisticRegression(penalty=penalty, solver='saga', max_iter=1000),
                    param_grid={'C': self.Cs},
                    scoring='neg_log_loss',
                    cv=3,
                    n_jobs=-1
                )
                clf.fit(X_embed_tune, y)
                losses.append((clf.best_score_, clf.best_params_['C']))
            avg_loss = sorted(losses, key=lambda x: x[0], reverse=True)
            best_Cs[penalty] = avg_loss[0][1]  # take the best average C

        print(f"Best C for L1: {best_Cs['l1']}, Best C for L2: {best_Cs['l2']}")

        # For each neuron, fit 10 models for each penalty with best C, keep best likelihood
        avg_log_likelihoods = {'l1': [], 'l2': []}
        for i in range(n_neurons):
            y = X_neural[:, i]
            for penalty in ['l1', 'l2']:
                C = best_Cs[penalty]
                best_ll = -np.inf
                for _ in range(self.num_models_per_neuron):
                    clf = LogisticRegression(penalty=penalty, C=C, solver='saga', max_iter=1000)
                    clf.fit(X_embed, y)
                    prob = clf.predict_proba(X_embed)
                    ll = -log_loss(y, prob, labels=[0, 1], normalize=True)
                    if ll > best_ll:
                        best_ll = ll
                avg_log_likelihoods[penalty].append(best_ll)

        data['avg_log_likelihoods'] = avg_log_likelihoods

        # Save to file
        output_file = Path("likelihoods_summary.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(avg_log_likelihoods, f)
        print(f"Saved average log-likelihoods to {output_file}")
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

    print(f"Running pipeline for container_id={container_id}, session={session}, stimulus={stimulus}")

    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),
        ImageToEmbeddingStep(embedding_cache_dir),
        AllenNeuralResponseExtractor(boc, eid_dict, stimulus_session_dict),
        AllenViTRegressionDatasetBuilder(),
        L1vRidgeGLMFitStep()
    ])
    import time
    start=time.time()
    result = pipeline.run((container_id, session, stimulus))
    end=time.time()
    print("Embed shape:", result['X_embed'].shape)
    print("Neural shape:", result['X_neural'].shape)
    print("L1 log-likelihoods:", result['avg_log_likelihoods']['l1'])
    print("Ridge log-likelihoods:", result['avg_log_likelihoods']['l2'])
    print(end-start)


class L1vRidgeGLMFitStep(PipelineStep):
    def __init__(self, Cs=np.logspace(-3, 1, 6), num_tune_neurons=10, num_models_per_neuron=10):
        self.Cs = Cs
        self.num_tune_neurons = num_tune_neurons
        self.num_models_per_neuron = num_models_per_neuron

    def process(self, data):
        X_embed = data['X_embed']
        X_neural = data['X_neural']
        n_neurons = X_neural.shape[1]

        neuron_indices = np.arange(n_neurons)
        rng = np.random.default_rng(seed=42)
        tune_indices = rng.choice(neuron_indices, size=min(self.num_tune_neurons, n_neurons), replace=False)

        # Tune C separately for L1 and L2
        best_Cs = {'l1': None, 'l2': None}
        for penalty in ['l1', 'l2']:
            losses = []
            for i in tune_indices:
                y = X_neural[:, i]
                clf = GridSearchCV(
                    LogisticRegression(penalty=penalty, solver='saga', max_iter=1000),
                    param_grid={'C': self.Cs},
                    scoring='neg_log_loss',
                    cv=5,
                    n_jobs=-1
                )
                clf.fit(X_embed, y)
                losses.append((clf.best_score_, clf.best_params_['C']))
            avg_loss = sorted(losses, key=lambda x: x[0], reverse=True)
            best_Cs[penalty] = avg_loss[0][1]  # take the best average C

        print(f"Best C for L1: {best_Cs['l1']}, Best C for L2: {best_Cs['l2']}")

        # For each neuron, fit 10 models for each penalty with best C, keep best likelihood
        avg_log_likelihoods = {'l1': [], 'l2': []}
        for i in range(n_neurons):
            y = X_neural[:, i]
            for penalty in ['l1', 'l2']:
                C = best_Cs[penalty]
                best_ll = -np.inf
                for _ in range(self.num_models_per_neuron):
                    clf = LogisticRegression(penalty=penalty, C=C, solver='saga', max_iter=1000)
                    clf.fit(X_embed, y)
                    prob = clf.predict_proba(X_embed)
                    ll = -log_loss(y, prob, labels=[0, 1], normalize=True)
                    if ll > best_ll:
                        best_ll = ll
                avg_log_likelihoods[penalty].append(best_ll)

        data['avg_log_likelihoods'] = avg_log_likelihoods
        print("Finished fitting all models.")
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
    import time
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

    print(f"Running pipeline for container_id={container_id}, session={session}, stimulus={stimulus}")

    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),
        ImageToEmbeddingStep(embedding_cache_dir),
        AllenNeuralResponseExtractor(boc, eid_dict, stimulus_session_dict),
        AllenViTRegressionDatasetBuilder(),
        L1vRidgeGLMFitStep()
    ])
    start=time.time()
    result = pipeline.run((container_id, session, stimulus))

    print("Embed shape:", result['X_embed'].shape)
    print("Neural shape:", result['X_neural'].shape)
    print("L1 log-likelihoods:", result['avg_log_likelihoods']['l1'])
    print("Ridge log-likelihoods:", result['avg_log_likelihoods']['l2'])
    end=time.time()
    print(end-start)

class L1GLMNeuralFitStep(PipelineStep):
    def __init__(self, save_path='glm_weights.npy', Cs=np.logspace(-3, 1, 6)):
        self.save_path = save_path
        self.Cs = Cs

    def process(self, data):
        X_embed = data['X_embed']
        X_neural = data['X_neural']
        models = []
        weight_matrix = np.zeros((X_neural.shape[1], X_embed.shape[1]))

        #for i in range(X_neural.shape[1]):
        for i in range(5):
            y = X_neural[:, i]
            clf = GridSearchCV(
                LogisticRegression(penalty='l1', solver='saga', max_iter=1000),
                param_grid={'C': self.Cs},
                scoring='neg_log_loss',
                cv=5,
                n_jobs=-1
            )
            clf.fit(X_embed, y)
            best_model = clf.best_estimator_
            models.append(best_model)
            weight_matrix[i, :] = best_model.coef_

        data['glm_models'] = models
        np.save(self.save_path, weight_matrix)
        print(f"Saved logistic regression weights to {self.save_path}")
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
    stimlus='natural_scenes'
    print(session, stimulus)
    print(f"Running pipeline for container_id={container_id}, session={session}, stimulus={stimulus}")

    pipeline = AnalysisPipeline([
        AllenStimuliFetchStep(boc),
        ImageToEmbeddingStep(embedding_cache_dir),
        AllenNeuralResponseExtractor(boc, eid_dict, stimulus_session_dict),
        AllenViTRegressionDatasetBuilder(),
        L1GLMNeuralFitStep(save_path='glm_weights_natural_images.npy')
    ])

    result = pipeline.run((container_id, session, stimulus))

    print("Embed shape:", result['X_embed'].shape)
    print("Neural shape:", result['X_neural'].shape)
    print(f"Fitted {len(result['glm_models'])} GLMs.")


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
        AllenViTRegressionDatasetBuilder()
    ])

    result = pipeline.run((container_id, session, stimulus))

    X_embed = result['X_embed']
    X_neural = result['X_neural']
    print("Embed shape:", X_embed.shape)
    print("Neural shape:", X_neural.shape)
'''