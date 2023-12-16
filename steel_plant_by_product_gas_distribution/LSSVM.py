"LSSVM Class"
import numpy as np
import pandas as pd
from steel_plant_by_product_gas_distribution.data_preprocessing import DataPreprocessing
import statsmodels.api as sm
import pickle
import steel_plant_by_product_gas_distribution.data_dictionary as dd
from typing import Tuple

class LSSVMVolatility:
    
    def __init__(self, sigma=2, c=1): 
        self.sigma:float = sigma
        self.c:int = c
        self.__x:np.array = None
        self.__y:np.array = None
        self.__alpha:np.array= None
        self.__b:float = None
    
    def _getKernelBlock(
        self, 
        x:np.array, 
        y:np.array
    ) -> np.array:
        sigma = self.sigma
        lam = 0.5/(sigma**2)
        c = self.c
        n = x.shape[0]
        m = self._getKernelMatrix(x,y)+ np.eye(n)/c
        return m

    def _getKernelMatrix(
        self,
        x:np.array, 
        y:np.array
    ) -> np.array:
        sigma = self.sigma
        lam = 0.5/(sigma**2)
        if x.ndim >1:
            x2 = sum((x**2).T).reshape(x.shape[0],1)
            y2 = sum((y**2).T).reshape(y.shape[0],1)
            norm = y2[:,...] + x2.T - 2*np.dot(y,x.T)
            return np.exp(-norm*lam)
        else:
            return np.exp(-(np.linalg.norm(x-y)**2)*(lam))

    def predict(
        self, 
        x_test:np.array
    ) -> np.array:
        sigma = self.sigma
        lam = 0.5/(sigma**2)
        alpha=self.__alpha
        b=self.__b
        c=self.c
        matrix = self._getKernelMatrix(self.__x,x_test)
        output = np.dot(matrix,alpha)+b
        return output.reshape(len(output))

    """
    This function computes the value of the function
       d(aT * K * a) / d x_i
    it is not the loss function of the W(alpha,x) yet
    """

    def _getParams(
        self, 
        x_sample:np.array,
        y_sample:np.array
    ) -> Tuple[np.array,float]:
        sigma = self.sigma
        lam = 0.5/(sigma**2)
        alpha=self.__alpha
        b=self.__b
        c=self.c
        n = y_sample.shape[0]
        matrix = self._getKernelBlock(x_sample,x_sample)
        temp_v = np.row_stack((np.ones(n), matrix))
        temp_h = np.row_stack(([0],np.ones((n,1))))
        A = np.column_stack((temp_h,temp_v ))
        temp_b = np.row_stack(([0],y_sample))
        params =  np.dot( np.linalg.pinv(A),temp_b )
        alpha = params[1:]
        b = params[0]
        return alpha,b
    
    def fit(
        self,
        X:np.array,
        y:np.array
    ):
        self.__x = np.array(X)
        self.__y = np.array(y)
        self.__alpha, self.__b = self._getParams(self.__x,self.__y)
    
    def getParameters(self) -> Tuple[np.array,float,np.array]:
        return self.__alpha, self.__b, self.__x
    
    def setParameters(
        self,
        alpha:float,
        b:float,
        x:np.array
    ):
        self.__alpha = alpha
        self.__b = b
        self.__x = x
