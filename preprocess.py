from re import T
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
import numpy as np  

pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)

SKLEARN_EMPTY_STEP = "passthrough"

def str_if_not_null(x):
    if pd.isnull(x) or (x is None) or pd.isna(x) or (x is not x) : 
        return x 
    return str(x)

class DataTypes_Auto_infer(BaseEstimator, TransformerMixin):
    """
    This class will infer data types automatically . 
    & 
    This also automatically deletes columns (values or same column name), removes rows where target variable is null and remove columns and rows where all records are null

    """

    def __init__(
        self, 
        target, 
        ml_usecase, 
        categorical_features = [], 
        numerical_features = [], 
        time_features = [],
        features_to_drop = [], 
        id_columns = [], 
        display_types = True ):
        """
        User to define the target (y) variable (we can't define it automatically) (SQL의 경우, id도 넣어줘야할 것 같은데?) 
        args : 
            target: 
            ml_usecase: 
            categorical_features: 
            numerical_features: 
            time_features: 
            features_to_drop:
            id_columns:
            display_types: 
        """
        self.target = target 
        self.ml_usecase = ml_usecase 
        self.features_to_drop = [str(x) for x in features_to_drop]
        self.categorical_features = [x for x in categorical_features if x not in self.features_to_drop]
        self.numerical_features = [x for x in numerical_features if x not in self.features_to_drop]
        self.display_types = display_types 
        self.id_columns = id_columns 

    def fit(self, dataset, y = None): # learning data types of all the columns 
        """
        args : 
            dataset: accepts a pandas dataframe 
        returns : 
            pandas dataframe
        
        """
        data = dataset.copy()

        # also make sure that all the column names are string 
        data.columns = [str(i) for i in data.columns]

        # drop any columns that were asked to drop 
        data.drop(columns = self.features_to_drop, axis=1, errors = "ignore",inplace= True)

        # if there are inf or -inf then replace them with NaN
        data.replace([np.inf, -np.inf],np.NaN, inplace=True)

        # can check if somehow everything is object, we can try converting them in float
        # object -> int64 
        # object -> datetime 추론
        # bool, categorical -> object   
        for col in data.select_dtypes(include = ['object']).columns:
            try:
                data[col] = data[col].astype("int64")
            except:
                None
        for col in data.select_dtypes(include = ['object']).drop(self.target, axis=1, errors= "ignore", inplace=False):
            try:
                data[col] = pd.to_datetime(data[col], infer_datetime_format= True, utc = False, errors ="raise")
            except:
                continue
        # if data type is bool or pandas Categorical, conver to categorical 
        for col in data.select_dtypes(include =['bool', 'category']).columns : 
            data[col] = data[col].astype("object")

        # with csv, if we have any null in a column that was int, pandas will read it as float 
        # so first we need to convert any such floats that have NaN and unique values lower than 20 




        return data

    def transform(self, dataset, y= None):
        """
        args :
            dataset : accept a pandas dataframe 
        returns :
            pandas dataframe
        """ 
        data =dataset.copy()

        return data 

    # fit_transform
    def fit_transform(self,dataset, y= None):
        data = dataset 
        return data 

