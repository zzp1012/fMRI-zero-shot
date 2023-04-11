import numpy as np
from sklearn.base import BaseEstimator

class ModelUtils:
    """Class for model utils. Mainly sklearn models."""
    
    @classmethod
    def auto(cls,
             model_name: str,
             **kwargs) -> BaseEstimator:
        """Auto load the model. Using scikit-learn Package.
        
        Args:
            model_name (str): Name of the model.
        
        Returns:
            model (nn.Module): The model.
        """
        if model_name == "mlp":
            from sklearn.neural_network import MLPRegressor
            return MLPRegressor(**kwargs)
        elif model_name == "lr":
            from sklearn.linear_model import LinearRegression
            return LinearRegression(**kwargs)
        elif model_name == "knn":
            from sklearn.neighbors import KNeighborsClassifier
            return KNeighborsClassifier(**kwargs)
        else:
            raise ValueError("model_name not found.")

    @classmethod
    def train(cls,
              model: BaseEstimator,
              X: np.ndarray,
              y: np.ndarray,) -> BaseEstimator:
        """Train the model.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
            y (np.ndarray): Labels. (N, )
        
        Returns:
            model (BaseEstimator): The trained model.
        """
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."

        model.fit(X, y)
        return model

    @classmethod
    def predict(cls,
                model: BaseEstimator,
                X: np.ndarray) -> np.ndarray:
        """Predict the labels.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
        
        Returns:
            y_pred (np.ndarray): Predicted labels. (N, )
        """
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."

        y_pred = model.predict(X)
        return y_pred
    
    @classmethod
    def score(cls,
              model: BaseEstimator,
              X: np.ndarray,
              y: np.ndarray) -> float:
        """Score the model.
        
        Args:
            model (BaseEstimator): The model.
            X (np.ndarray): Features. (N, D)
            y (np.ndarray): Labels. (N, )

        Returns:
            score (float): The score.
        """
        # simple check
        assert len(X.shape) == 2, \
            "X should be 2D."
    
        score = model.score(X, y)
        return score