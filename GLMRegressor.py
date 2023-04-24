#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class GLMRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self, family, **glm_kwargs):        
        self.family = family
        self.glm_kwargs = glm_kwargs
        self.glm_model = None
        
    def fit(self, X, y):
        self.glm_model = GLM(y, X, family=self.family, **self.glm_kwargs)
        self.model_fit = self.glm_model.fit()
        
        return self
    
    def predict(self, X):
                
        return self.model_fit.predict(X) 


# In[ ]:




