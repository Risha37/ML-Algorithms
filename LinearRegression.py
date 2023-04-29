"""
        Description
        ----------
        
        Key Parameters
        ----------
        -
        
        Optional Parameters
        ----------
        -
        
        Returns
        ----------
        -
"""
        
        
        
"""
$ Author: Risha $
$ https://github.com/Risha37 $
$ Revision: 1.0 $

TODO
- Add Regularizations Terms 'regularization_term'
- Add different Gradient Descent Algorithms (Stochastic, Mini-Batch)
"""


import numpy as np


class LinearRegression():
    
    def __init__(self):
        """
        Description
        ----------
        The following are a set of methods intended for regression in which
        the target value is expected to be a linear combination of the features.
        """
        pass
    
    
    def fit(self,
            X,
            y
           ):
        """
        Description
        ----------
        Fits a linear model using the Normal Equation.
        
        Parameters
        ----------
        - X : array-like of shape (n_samples, n_features)
            Training data
            
        - y : array-like of shape (n_samples, 1) or (n_samples, n_targets)
            Target data

        Returns
        ----------
        - Learned weights : array-like of shape (n_features+1, 1)
            Coefficients, Intercept
        """
        #Add ones column vector to represent x_0
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        return (np.linalg.pinv(X.T @ X) @ (X.T @ y))
    
    
    def predict(self,
                X,
                optimal_weights
               ):
        """
        Description
        ----------
        Predicts a target given the features value and the learned weights.
        
        Parameters
        ----------
        - X : array-like of shape (n_samples, n_features)
            Testing data
        
        - optimal_weight : array-like of shape (n_features+1, 1)
            Learned coefficients & intercept after training
            
        Returns
        ----------
        - y predict (y hat) : array-like of shape (n_samples, 1) or (n_samples, n_targets)
            Targets predicted using the 'optimal' weights
        """
        return ((X @ optimal_weights[:-1]) + optimal_weights[-1])
    
    
    def GradientDescent(self,
                        X,
                        y,
                        learning_rate: float=0.1,
                        epochs: int=20,
                        random_state: int=42,
                        initial_weight=None,
                        regularization_term=None,
                        return_loss: bool=False,
                        print_results_epoch=[False, None]
                       ):
        """
        Description
        ----------
        Fits a linear model using the Batch Gradient Descent Algorithm.
        
        Key Parameters
        ----------
        - X : array-like of shape (n_samples, n_features)
            Training data
        
        - y : array-like of shape (n_samples, 1) or (n_samples, n_targets)
            Target data
            
        Optional Parameters
        ----------
        - learning_rate : float, default=0.1
            Controls how quickly the model is adapted to the problem (alpha)
        
        - epochs : int, default=20
            The maximum number of passes over the training data
            
        - random_state : int, default=42
            Set random seed to keep consistent results
            
        - initial_weight : array-like of shape (n_features+1, 1), default=None
            [Coefficients, Intercept] Custom initialized weights instead of random initialization.
            
        - (WIP)regularization_term : {"Lasso", "Ridge", "ElasticNet", None}, default=None
            Imposing a penalty on the size of the coefficients to prevent overfitting
            
        - return_loss : bool, default=False
        
        - print_results_epoch: list [bool, int], default=None
            Print results every n epochs
            
        Returns
        ----------
        - W : array-like of shape (n_features+1, 1)
            Learned Coefficients & Intercept
            
        - loss : array-like of shape (1, 1), optional
            The final loss after n epochs of training
        """
        np.random.seed(random_state)
        
        #Initialise random weights
        W = np.random.randn(X.shape[1]+1, 1) if initial_weight is None else initial_weight
        
        #Add ones column vector to represent x_0
        X = np.append(X, np.ones((len(X), 1)), axis=1)
        
        m = len(X)
        for epoch in range(epochs):
            if return_loss:
                #Vectorized form MSE Cost Function
                J = (1/m) * ((X @ W) - y).T @ ((X @ W) - y)
            #Vectorized form gradient of the cost function
            dJ = (1/m) * (X.T @ (X @ W - y))
            
            #Gradient Descent step
            W = W - learning_rate * dJ
            
            if print_results_epoch[0]:
                if epoch % print_results_epoch[1] == 0 and return_loss:
                    print(f"Epoch: {epoch} | Loss: {J} |\nWeights: \n{W}")
                elif epoch % print_results_epoch[1] == 0 and not return_loss:
                    print(f"Epoch: {epoch} |\nWeights: \n{W}")
        
        return (W) if not return_loss else (W, J)
    
    
    def score(self,
              y_true,
              y_predict,
              criteria: str="R2"
             ):
        """
        Description
        ----------
        (Evaluation metrics) Calculates the {R squared} score.
        
        Key Parameters
        ----------
        - y_true: array-like of shape (n_samples,)
            The ground truth target data.
            
        - y_predict: array-like of shape (n_samples,)
            The predicted target data.
        
        Optional Parameters
        ----------
        - criteria: {"MSE", "RMSE", "MAE", "R2"}, default='R2'
            The criteria used to evaluate the model.
            
        Returns
        ----------
        - score: float
            The R2 score between y_true and y_predict.
        """
        if criteria == 'R2':
            ss_res = np.sum((y_true - y_predict)**2)
            ss_tot = np.sum((y_true - np.mean(y_true))**2)
            score = 1 - (ss_res / ss_tot)
        elif criteria == 'MSE':
            score = np.mean((y_true - y_predict)**2)
        elif criteria == 'RMSE':
            score = np.sqrt(np.mean((y_true - y_predict)**2))
        elif criteria == 'MAE':
            score = np.mean(np.abs(y_true - y_predict))
        
            
        return score



from DataProcessing import DataProcessing

lin_reg = LinearRegression()
data_proc = DataProcessing()


np.random.seed(42)

X = np.random.rand(100, 2)
y = (7 * X[:, 0] + 3 * X[:, 1] + 9).reshape(-1, 1) + np.random.randn(100, 1)

X_train, X_test, y_train, y_test = data_proc.train_test_split(X, y)


opt_w0 = lin_reg.fit(X_train, y_train)
opt_w1 = lin_reg.GradientDescent(X_train, y_train, epochs=200)

print(f"Optimal weights (sould be close or equal to W=[7, 3] b=[9])\nNormal Equation = {opt_w0}\nBatch Gradient Descent = {opt_w1}")


y_pred0 = lin_reg.predict(X_test, opt_w0)
y_pred1 = lin_reg.predict(X_test, opt_w1)

score0 = lin_reg.score(y_test, y_pred0)
score1 = lin_reg.score(y_test, y_pred1)

print(f"The Score for the Normal Equation Model = {score0} || Gradient Descent Model = {score1}")