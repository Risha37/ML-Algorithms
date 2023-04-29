"""
$ Author: Risha $
$ https://github.com/Risha37 $
$ Revision: 1.0 $

TODO
- Add random shuffling into data splitting function 'train_test_split()'
"""


import numpy as np


class DataProcessing():
    
    def __init__(self):
        """
        Description
        ----------
        
        """
        pass
    
    
    def train_test_split(self,
                         X,
                         y,
                         test_size: float=0.2
                        ):
        """
        Description
        ----------
        Splits the data into training and testing sets (TODO random shuffling)
        
        Key Parameters
        ----------
        - X : array-like of shape (n_samples, n_features)
            Features data
        
        - y : array-like of shape (n_samples, 1) or (n_samples, n_targets)
            Target data
            
        Optional Parameters
        ----------
        - test_size : float, default=0.2
            The proportion of the dataset to include in the test split
        
        Returns
        ----------
        - X_train : array-like of shape (whole_date - test_size, n_features)
            Training set of features
        
        - X_test : array-like of shape (test_size, n_features)
            Testing set of features
        
        - y_train : array-like of shape (whole_date - test_size, 1) or (whole_date - test_size, n_targets)
            Training set of target values
        
        - y_test : array-like of shape (test_size, n_features) or (test_size, n_targets)
            Testing set of target values
        """
        split_size = int(len(X) - int(test_size * len(X)))
            
        X_train, y_train = X[:split_size], y[:split_size]
        X_test, y_test = X[split_size:], y[split_size:]
        
        return (X_train, X_test, y_train, y_test)