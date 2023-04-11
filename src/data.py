import os
import numpy as np
import pickle
from typing import Tuple, List

class fMRIDataUtils:
    """Data utils class for fMRI data.
    """

    @classmethod
    def read(cls,
             person_id: int,
             root: str="../data") -> Tuple[dict, dict]:
        """read the raw data from the data folder.

        Args:
            person_id (int): the person id.
            root (str, optional): the root path of the data folder. Defaults to "../data".
        
        Returns:
            word2inputs (dict): {key: [v1, v2, v3, v4, v5, v6]}. v - (1, D) np.ndarray, 60 keys in total.
            word2features (dict): {key: v}. v - (D, ) np.ndarray, 60 keys in total.
        """
        assert person_id in range(1, 10), \
            "person_id should be in range [1, 9]."
        # load the word2inputs
        with open(os.path.join(root, f"data_science/P{person_id}.pkl"), "rb") as f:
            word2inputs = pickle.load(f)

        # load the word2features
        with open(os.path.join(root, f"word2features.pkl"), "rb") as f:
            word2features = pickle.load(f)
        
        print(word2features.keys())
      
        return word2inputs, word2features
    
    @classmethod
    def split(cls,
              inputs: dict,
              features: dict,
              leave_out: List[str],) -> Tuple[dict, dict, dict, dict]:
        """split the data into train, test set by leave out some words.

        Args:
            inputs (dict): the inputs.
            features (dict): the features.
            leave_out (List[str]): the words to leave out.
        
        Returns:
            train_inputs (dict): the train inputs.
            train_features (dict): the train features.
            test_inputs (dict): the test inputs.
            test_features (dict): the test features.
        """
        assert inputs.keys() == features.keys(), \
            "the keys of inputs and features should be the same."
        assert len(leave_out) > 0 and len(leave_out) < len(inputs.keys()), \
            "the length of leave_out should be in range [1, 59]."

        train_inputs, train_features = {}, {}
        test_inputs, test_features = {}, {}
        for key, _ in inputs.items():
            if key in leave_out:
                test_inputs[key] = inputs[key]
                test_features[key] = features[key]
            else:
                train_inputs[key] = inputs[key]
                train_features[key] = features[key]

        return train_inputs, train_features, test_inputs, test_features
    
    @classmethod
    def to_regression(cls,  
                      inputs: dict,
                      features: dict,) -> Tuple[np.ndarray, np.ndarray]:
        """convert the data to regression format.

        Args:
            inputs (dict): the inputs.
            features (dict): the features.
        
        Returns:
            X (np.ndarray): the inputs in regression format. (N, D)
            y (np.ndarray): the features in regression format. (N, 25)
        """
        assert inputs.keys() == features.keys(), \
            "the keys of inputs and features should be the same."
        
        X, y = [], []
        for key, input_lst in inputs.items():
            for input_ in input_lst:
                X.append(input_)
                y.append(features[key])

        X = np.concatenate(X, axis=0)
        y = np.array(y)
        return X, y
    
    @classmethod
    def to_classification(cls,
                          features: dict,) -> Tuple[np.ndarray, np.ndarray]:
        """convert the data to classification format.

        Args:
            features (dict): the features.
        
        Returns:
            X (np.ndarray): the inputs in classification format. (N, 25)
            y (np.ndarray): the labels in classification format. (N, )
        """

        X, y = [], []
        for key, feature in features.items():
            # normalize the feature
            X.append(feature / np.linalg.norm(feature))
            y.append(key)

        X = np.array(X)
        y = np.array(y)
        return X, y
