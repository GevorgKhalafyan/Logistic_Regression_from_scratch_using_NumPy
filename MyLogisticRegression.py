import numpy as np


class LogisticMulticlass:
    def __init__(self, lr=0.005, max_iter=1000, C=None):
        self.lr = lr
        self.max_iter = max_iter
        if C is not None:
            self.C = C
        else:
            self.C = float("inf")
        self.losses = np.array([])
        
    @staticmethod
    def _onehot(labels):
        """
        We want our model to work like a library one.
        Therefore, the target variable must be one-dimensional.
        But in the implementation, we need the target variable in the form of one-hot.
        That is exactly what this function does.
        """
        dict_onehot = {key: value for value, key
                                           in enumerate(np.unique(labels))}
        labels_onehot = np.array(list(map(dict_onehot.get, labels)))
        result_onehot = np.squeeze(np.eye(len(dict_onehot))
                                               [labels_onehot.reshape(-1)])
        
        return result_onehot
    
    def _calc_probs(self, data_bias):
        probs = (np.exp(-np.dot(data_bias, self.w)).T /
                     np.sum(np.exp(-np.dot(data_bias, self.w)), axis=1)).T
        
        return probs
    
    def _calc_loss(self, data_bias, labels_encoded):
        loss = (
            np.trace(
                np.dot(data_bias, np.dot(self.w,np.transpose(labels_encoded)))
            )
            + np.sum(
                np.log(np.sum(np.exp(np.dot(-data_bias, self.w)), axis=1))
            )
        ) / data_bias.shape[0]
        
        return loss
    
    def _calc_grad(self, data_bias, labels_encoded, probabilities):
        grad = (
            np.dot(np.transpose(data_bias), labels_encoded -  probabilities)
            + self.w / self.C
        ) / data_bias.shape[0]

        return grad
        
    
    def fit(self, data, labels):
        """
        This function trains the model through gradient descent.
        """
        labels_encoded = self._onehot(labels)
        data_bias = np.concatenate((data, np.ones((len(data), 1))), axis = 1)
    
        self.w = np.random.randn(np.size(data, 1) + 1,
                                 np.size(labels_encoded, 1))
        # Training loop
        for i in range(self.max_iter):
            probs = self._calc_probs(data_bias) # calculating probabilities    
            
            loss = self._calc_loss(data_bias, labels_encoded) # calculating loss
            
            grad = self._calc_grad(data_bias, labels_encoded, probs) # calculating gradients
            
            self.w += -self.lr * grad # updating model parameters
            
            self.losses = np.append(self.losses, loss)
            
    
    def predict(self, data):
        """
        This function makes predictions.
        """
        data_bias = np.concatenate((data, np.ones((len(data), 1))), axis = 1)
        
        probs = (np.exp(-np.dot(data_bias, self.w)).T /
             np.sum(np.exp(-np.dot(data_bias, self.w)), axis=1)).T
        labels_pred = probs.argmax(axis=1)
        
        return labels_pred