import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):

        # Things the model will learn during training
        self.weights = None
        self.bias = None

        # Normalization parameters
        self.X_min_values = None
        self.X_max_values = None

        # Hyperparameters
        self.lr = lr
        self.n_iterations = n_iterations

        # History of the loss function during training
        self.loss_history = []

    def fit_preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalized the features with min-max normalization. Do not include y in this function, only X.
        """
        self.X_min_values = X.min()
        self.X_max_values = X.max()

        if (self.X_max_values - self.X_min_values).eq(0).any():
            raise ValueError("Cannot perform min-max normalization when max and min values are equal for any feature.")
        
        return (X - self.X_min_values) / (self.X_max_values - self.X_min_values)


    def preprocess(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalized the features with min-max normalization that is already fitted on the training data.
        """
        if self.X_min_values is None or self.X_max_values is None:
            raise ValueError("Preprocessing has not been fitted.")

        if (self.X_max_values - self.X_min_values).eq(0).any():
            raise ValueError("Cannot perform min-max normalization when max and min values are equal for any feature.")

        return (X - self.X_min_values) / (self.X_max_values - self.X_min_values)

    def inverse_transform_x(self, x: float):
        """
        Inverse transform a single feature value to its original scale.
        """
        return x * (self.X_max_values - self.X_min_values) + self.X_min_values

    def _calculate_loss(self, error):
        """
        Calculates the mean squared error loss
        
        Args:
            y_true (array<m>): a vector of floats
            y_pred (array<m>): a vector of floats
            
        Returns:
            A float representing the mean squared error loss
        """
        return np.mean((error) ** 2)

    def fit(self, X: pd.DataFrame, y):
        """
        Estimates parameters for the regression model
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        self.loss_history = []

        self.weights = np.ones(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias

            error = predictions - y

            mse = self._calculate_loss(error)
            self.loss_history.append(mse)

            dw, db = self._compute_gradients(X, error)
            self._update_parameters(dw, db)

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions
        
        Note: should be called after .fit(). Assumes X has not been preprocessed.
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been fitted.")

        X = self.preprocess(X)
        return np.dot(X, self.weights) + self.bias

    def _compute_gradients(self, X, error):
        """
        Computes the gradients of the loss with respect to weights and bias
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            error (array<m>): a vector of floats representing the error between predictions and true values
            
        Returns:
            dw (array<n>): a vector of floats representing the gradient of the loss with respect to weights
            db (float): a float representing the gradient of the loss with respect to bias
        """
        m = X.shape[0]
        dw = (2/m) * np.dot(X.T, error)
        db = (2/m) * np.sum(error)
        return dw, db


    def _update_parameters(self, dw, db):
        """
        Updates the model parameters using the gradients and learning rate
        
        Args:
            dw (array<n>): a vector of floats representing the gradient of the loss with respect to weights
            db (float): a float representing the gradient of the loss with respect to bias
        """
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
        return



class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.fit is not implemented yet.")
    
    def predict_proba(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict_proba is not implemented yet.")
        
    def predict(self, X):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.predict is not implemented yet.")
    
    def sigmoid(self, z):
        # ====================================
        # YOUR CODE GOES HERE
        # ====================================
        raise NotImplementedError("LogisticRegression.sigmoid is not implemented yet.")