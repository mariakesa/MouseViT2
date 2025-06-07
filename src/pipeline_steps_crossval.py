from pipeline_steps_in_sample import AllenStimuliFetchStep, PipelineStep, ImageToEmbeddingStep, AnalysisPipeline, make_container_dict
from sklearn.model_selection import GroupKFold
from dotenv import load_dotenv
from data_loader import allen_api
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.proportion import proportion_confint
from collections import defaultdict
from skbio.stats.composition import clr
from joblib import Parallel, delayed

class StimulusGroupKFoldSplitterStep(PipelineStep):
    def __init__(self, boc, eid_dict, stimulus_session_dict, n_splits=10):
        """
        :param boc: Allen BrainObservatoryCache
        :param eid_dict: container_id -> { session: eid }
        :param stimulus_session_dict: e.g. {'three_session_A': [...], ...}
        :param n_splits: how many CV folds
        """
        self.boc = boc
        self.eid_dict = eid_dict
        self.stimulus_session_dict = stimulus_session_dict
        self.n_splits = n_splits

    def process(self, data):
        """
        data requires 'container_id', 'session', 'stimulus'.
        Creates data['folds'] => list of (X_train, frames_train, X_test, frames_test).
        """
        container_id = data['container_id']
        session = data['session']
        stimulus = data['stimulus']
        
        valid_stims = self.stimulus_session_dict.get(session, [])
        if stimulus not in valid_stims:
            raise ValueError(f"Stimulus '{stimulus}' not valid for session '{session}'. "
                             f"Valid: {valid_stims}")

        session_eid = self.eid_dict[container_id][session]

        dataset = self.boc.get_ophys_experiment_data(session_eid)
        
        #dff_traces = dataset.get_dff_traces()[1]  # shape (n_neurons, n_timepoints)
        dff_traces= self.boc.get_ophys_experiment_events(ophys_experiment_id=session_eid)
        #dff_traces = dataset

        stim_table = dataset.get_stimulus_table(stimulus)
        print(stim_table)


        X_list, frame_list, groups = [], [], []

        for _, row_ in stim_table.iterrows():
            if row_['frame']!=-1:
                start_t, end_t = row_['start'], row_['end']
                frame_idx = row_['frame']
                time_indices = range(start_t, end_t)

                if len(time_indices) == 0:
                    trial_vector = np.zeros(dff_traces.shape[0])
                else:
                    relevant_traces = dff_traces[:, time_indices]
                    #trial_vector = np.max(relevant_traces, axis=1)
                    threshold = 0.0  # or pick something domain-appropriate
                    trial_vector = np.max(relevant_traces, axis=1)

                    # Convert to binary: 1 if above threshold, else 0
                    trial_vector = (trial_vector > threshold).astype(float)
                
                X_list.append(trial_vector)
                frame_list.append(frame_idx)
                groups.append(frame_idx)
            else:
                pass

        X = np.vstack(X_list)
        print(X.shape)
        frames = np.array(frame_list)
        groups = np.array(groups)

        folds = []
        gkf = GroupKFold(n_splits=self.n_splits)
        for train_idx, test_idx in gkf.split(X, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx]
            frames_train, frames_test = frames[train_idx], frames[test_idx]
            folds.append((X_train, frames_train, X_test, frames_test))

        data['folds'] = folds
        return data


class MergeEmbeddingsStep(PipelineStep):
    """
    Reads the embedding file from data['embedding_file'],
    merges it with each fold in data['folds'], resulting in data['merged_folds'].
    """

    def __init__(self):
        # If you prefer, you can pass an argument here, e.g. `embedding_file`, 
        # but in this design, we read it from data.
        pass

    def process(self, data):
        """
        We expect:
          data['embedding_file'] -> path to a pickle file containing a dict: {stim_name: 2D array of embeddings}
          data['folds'] -> list of (X_train, frames_train, X_test, frames_test)
          data['stimulus'] -> e.g. 'natural_movie_one'
        
        We'll create data['merged_folds'] = list of (Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test).
        """
        embedding_file = data['embedding_file']
        stimulus = data['stimulus']
        folds = data['folds']

        # Load embeddings
        with open(embedding_file, 'rb') as f:
            all_stim_embeddings = pickle.load(f)

        # e.g. shape (#frames_in_stim, embedding_dim)
        # Note: we assume the indexing in all_stim_embeddings[stimulus]
        # matches the 'frame_idx' from the Allen table.
        embed_array = all_stim_embeddings[stimulus]
        #print(embed_array.shape)

        merged_folds = []
        for (Xn_train, frames_train, Xn_test, frames_test) in folds:
            # Build Xe_train from embed_array
            Xe_train = np.array([embed_array[f_idx] for f_idx in frames_train], dtype=np.float32)
            Xe_test  = np.array([embed_array[f_idx] for f_idx in frames_test], dtype=np.float32)

            merged_folds.append((Xn_train, Xe_train, Xn_test, Xe_test, frames_train, frames_test))

        data['merged_folds'] = merged_folds
        return data

class ConfidenceIntervalValidationStep(PipelineStep):
    def __init__(self, C=0.001, alpha=0.05, n_jobs=-1):
        self.C = C
        self.alpha = alpha
        self.n_jobs = n_jobs

    def process(self, data):
        merged_folds = data['merged_folds']
        n_neurons = merged_folds[0][0].shape[1]
        rng = np.random.default_rng(seed=42)

        def process_neuron(i_neuron):
            fold_accuracies_real = []
            fold_accuracies_perm = []

            for (Xn_train, Xe_train_raw, Xn_test, Xe_test_raw, frames_train, frames_test) in merged_folds:
                try:
                    # Memory-efficient CLR (cast to float32 early)
                    Xe_train_raw = Xe_train_raw.astype(np.float32)
                    Xe_test_raw = Xe_test_raw.astype(np.float32)

                    Xe_train = clr(Xe_train_raw + 1e-6)
                    Xe_test = clr(Xe_test_raw + 1e-6)

                    Xe_train_perm = Xe_train[rng.permutation(Xe_train.shape[0])]

                    y_train = Xn_train[:, i_neuron]
                    y_test = Xn_test[:, i_neuron]

                    if len(np.unique(y_train)) < 2:
                        continue  # not enough class variation

                    # Fit on real
                    model_real = LogisticRegression(penalty='l2', C=self.C, solver='saga',
                                                    max_iter=1000, verbose=0)
                    model_real.fit(Xe_train, y_train)
                    pred_real = model_real.predict_proba(Xe_test)[:, 1]
                    del model_real  # Free memory

                    # Fit on permuted
                    model_perm = LogisticRegression(penalty='l2', C=self.C, solver='saga',
                                                    max_iter=1000, verbose=0)
                    model_perm.fit(Xe_train_perm, y_train)
                    pred_perm = model_perm.predict_proba(Xe_test)[:, 1]
                    del model_perm

                    frame_to_obs = defaultdict(list)
                    frame_to_pred_real = defaultdict(list)
                    frame_to_pred_perm = defaultdict(list)

                    for j, f in enumerate(frames_test):
                        frame_to_obs[f].append(y_test[j])
                        frame_to_pred_real[f].append(pred_real[j])
                        frame_to_pred_perm[f].append(pred_perm[j])

                    correct_real = 0
                    correct_perm = 0
                    total = 0

                    for f in frame_to_obs:
                        obs = frame_to_obs[f]
                        count = sum(obs)
                        n = len(obs)
                        if n < 2:
                            continue

                        ci_low, ci_upp = proportion_confint(count, n, alpha=self.alpha, method='wilson')

                        correct_real += sum(ci_low <= p <= ci_upp for p in frame_to_pred_real[f])
                        correct_perm += sum(ci_low <= p <= ci_upp for p in frame_to_pred_perm[f])
                        total += len(obs)

                    if total > 0:
                        fold_accuracies_real.append(correct_real / total)
                        fold_accuracies_perm.append(correct_perm / total)

                    # Free per-fold memory
                    del Xe_train_raw, Xe_test_raw, Xe_train, Xe_test
                    del Xe_train_perm, y_train, y_test
                    del frame_to_obs, frame_to_pred_real, frame_to_pred_perm

                except Exception:
                    continue

            acc_real = np.mean(fold_accuracies_real) if fold_accuracies_real else np.nan
            acc_perm = np.mean(fold_accuracies_perm) if fold_accuracies_perm else np.nan
            return acc_real, acc_perm

        results = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(process_neuron)(i) for i in range(n_neurons)
        )

        accuracies_real = np.array([r[0] for r in results])
        accuracies_perm = np.array([r[1] for r in results])

        data['ci_accuracy_real'] = accuracies_real
        data['ci_accuracy_perm'] = accuracies_perm

        with open("ci_accuracy_comparison_0.001.pkl", "wb") as f:
            pickle.dump({
                'real': accuracies_real,
                'perm': accuracies_perm
            }, f)

        print("Saved confidence interval accuracy comparison to ci_accuracy_comparison.pkl")
        return data

class CrossValEventOnlyLogLikelihoodStep(PipelineStep):
    def __init__(self, C=0.001, n_jobs=-1):
        self.C = C
        self.n_jobs = n_jobs

    def process(self, data):
        merged_folds = data['merged_folds']
        n_neurons = merged_folds[0][0].shape[1]

        def process_neuron(i_neuron):
            real_lls, perm_lls = [], []

            for (Xn_train, Xe_train_raw, Xn_test, Xe_test_raw, _, _) in merged_folds:
                try:
                    # Normalize embeddings
                    Xe_train = clr(Xe_train_raw + 1e-6).astype(np.float32)
                    Xe_test = clr(Xe_test_raw + 1e-6).astype(np.float32)
                    Xe_train_perm = np.random.permutation(Xe_train)

                    y_train = Xn_train[:, i_neuron]
                    y_test = Xn_test[:, i_neuron]

                    if len(np.unique(y_train)) < 2:
                        continue

                    # Train real model
                    model_real = LogisticRegression(penalty='l2', C=self.C, solver='saga', max_iter=1000)
                    model_real.fit(Xe_train, y_train)
                    prob_real = model_real.predict_proba(Xe_test)[:, 1]

                    # Train permuted model
                    model_perm = LogisticRegression(penalty='l2', C=self.C, solver='saga', max_iter=1000)
                    model_perm.fit(Xe_train_perm, y_train)
                    prob_perm = model_perm.predict_proba(Xe_test)[:, 1]

                    # Evaluate only on spikes (y == 1)
                    mask = y_test == 1
                    if not np.any(mask):
                        continue

                    ll_real = np.mean(np.log(prob_real[mask] + 1e-8))
                    ll_perm = np.mean(np.log(prob_perm[mask] + 1e-8))

                    real_lls.append(ll_real)
                    perm_lls.append(ll_perm)
                except Exception:
                    continue

            # Average across folds
            avg_real = np.mean(real_lls) if real_lls else float('-inf')
            avg_perm = np.mean(perm_lls) if perm_lls else float('-inf')
            return avg_real, avg_perm

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_neuron)(i) for i in range(n_neurons)
        )

        real_lls = np.array([r[0] for r in results])
        perm_lls = np.array([r[1] for r in results])

        data['event_only_loglik_real'] = real_lls
        data['event_only_loglik_perm'] = perm_lls

        with open("event_only_loglik_cv_0.001.pkl", "wb") as f:
            pickle.dump({
                'real': real_lls,
                'perm': perm_lls,
                'delta': real_lls - perm_lls
            }, f)

        print("Saved event-only log-likelihoods (cross-validated) to event_only_loglik_cv.pkl")
        return data




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
        StimulusGroupKFoldSplitterStep(boc, eid_dict, stimulus_session_dict),
        MergeEmbeddingsStep(),
        #ConfidenceIntervalValidationStep()
        CrossValEventOnlyLogLikelihoodStep()
    ])
    import time
    start=time.time()
    result = pipeline.run((container_id, session, stimulus))
    end=time.time()
    print('Time taken', end-start)

