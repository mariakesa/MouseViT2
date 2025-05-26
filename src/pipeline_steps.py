from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pickle
import torch
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
import numpy as np
import pickle
from pathlib import Path
from pathlib import Path
from transformers import AutoProcessor, AutoModelForImageClassification
from torch.nn.functional import softmax


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""
    
    @abstractmethod
    def process(self, data):
        """Process the input data and return the transformed output."""
        pass

class AllenStimuliFetchStep(PipelineStep):
    """
    Fetches data from the Allen Brain Observatory.
    The session IDs are hard-coded since the stimuli are always the same.
    """
    # Hard-coded sessions
    SESSION_A = 501704220
    SESSION_B = 501559087
    SESSION_C = 501474098

    def __init__(self, boc):
        """
        :param boc: BrainObservatoryCache instance (via the AllenAPI singleton).
        """
        self.boc = boc

    def process(self, data):
        """
        Expects data to be either None or have (container_id, session, stimulus).
        We fetch a dictionary of raw stimuli arrays, store them in data['raw_data_dct'].
        """
        if isinstance(data, tuple):
            container_id, session, stimulus = data
            data = {'container_id': container_id, 'session': session, 'stimulus': stimulus}
        elif data is None:
            data = {}

        raw_data_dct = {}

        movie_one_dataset = self.boc.get_ophys_experiment_data(self.SESSION_A)
        raw_data_dct['natural_movie_one'] = movie_one_dataset.get_stimulus_template('natural_movie_one')

        movie_two_dataset = self.boc.get_ophys_experiment_data(self.SESSION_C)
        raw_data_dct['natural_movie_two'] = movie_two_dataset.get_stimulus_template('natural_movie_two')

        movie_three_dataset = self.boc.get_ophys_experiment_data(self.SESSION_A)
        raw_data_dct['natural_movie_three'] = movie_three_dataset.get_stimulus_template('natural_movie_three')

        natural_images_dataset = self.boc.get_ophys_experiment_data(self.SESSION_B)
        raw_data_dct['natural_scenes'] = natural_images_dataset.get_stimulus_template('natural_scenes')

        data['raw_data_dct'] = raw_data_dct
        return data
    
class ImageToEmbeddingStep(PipelineStep):
    """
    Converts images into softmax class probability vectors using a specified Hugging Face model,
    checking if a cached version already exists, and saves to disk if not.
    """

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
        """
        Convert each frame to softmax probabilities over classes.
        Returns shape (n_frames, n_classes)
        """
        n_frames = len(frames_array)
        frames_3ch = np.repeat(frames_array[:, None, :, :], 3, axis=1)  # make 3-channel

        with torch.no_grad():
            # Get class count from one forward pass
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
