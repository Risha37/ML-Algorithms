"""
WIP
$ Author: Risha $
$ https://github.com/Risha37 $
$ Revision: 1.0 $

TODO
- 
"""


import numpy as np


class LogisticRegression():
    
    def __init__(self):
        """
        Description
        ----------
        
        """
        pass
    
    
    def sigmoid(self,
                x
               ):
        """
        Description
        ----------
        
        Parameters
        ----------
        -
        
        Returns
        ----------
        -
        """
        return 1/(1+np.exp(-x))
    
    
    def fit(self,
            X,
            y,
            learning_rate: float=0.1,
            epochs: int=20,
            random_state: int=42
           ):
        """
        Description
        ----------
        
        Parameters
        ----------
        -
        
        Returns
        ----------
        -
        """
        np.random.seed(random_state)
        
        #Initialise random weights
        W = np.random.randn(X.shape[1]+1, 1) if initial_weight is None else initial_weight
        
        #Add ones column vector to represent x_0
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        
        
        for epoch in range(epochs):
            
            z = X @ W
            h_x = self.sigmoid(z)
            
            J = (-1/m) * np.sum((y @ np.log(h_x)) + ((1-y) @ np.log(1-h_x)))