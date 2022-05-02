from re import T
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
from IPython.display import display
import ipywidgets as wg   
from ipywidgets import Layout 
import sys 
from utils import infer_ml_usecase

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
        self.time_features = [x for x in time_features if x not in self.features_todrop]

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
        # int & unique_count>2 -> float32 
        #   
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
        for col in data.select_dtypes(include = ['float64']).columns:
            data[col] = data[col].astype("float32")
            # count how many Nas are there 
            na_count = sum(data[col].isnull())

            # count how many digits are there that have decimiles
            count_float = np.nansum([False if r.is_integer() else True for r in data[col]])
            # total decimals digit 
            count_float = (count_float - na_count)
            if (count_float == 0) & (data[col].nunique() <= 20) & (na_count>0):
                data[col] = data[col].astype("object")
                
        # if column is int and unique_counts are more than two, then (exclude target):
        for col in data.select_dtypes(include= ["int64"]).columns:
            if col != self.target:
                if data[col].nunique()<=20:
                    data[col] = data[col].apply(str_if_not_null)
                    # null 이 아닌 경우 string, null => int64
                else: 
                    data[col] = data[col].astype("float32")
        
        # now in case, we were given any specific columns dtypes in advance, we will over ride 
        for col in self.categorical_features:
            try: 
                data[col] = data[col].astype(str_if_not_null)
            except: 
                data[col] = dataset[col].astype(str_if_not_null) 
        
        for col in self.numerical_features:
            try: 
                data[col] = data[col].astype("float32")
            except: 
                data[col] = dataset[col].astype("float32")
        
        for col in self.time_features:
            try:
                data[col] = pd.to_datetime(data[col],infer_datetime_format= True, utc = False, errors ="raise")
            except: 
                data[col] = pd.to_datetime(dataset[col],infer_datetime_format=True, uec = False, errors = "raise")
        
        for col in data.select_dtypes(include=["datetime64", "datetime64[ns, UTC]"]).columns:
            data[col] = data[col].astype("datetime64[ns]")

        self.learnd_dtypes = data.dtypes
        self.training_columns = data.drop(self.target, axis=1).columns
        # 1. inf 값 처리하기 
        # if there are inf or -inf then replace them with NaN 
        data = data.replace([np.inf,-np.inf], np.NaN).astype(self.learnd_dtypes)

        # 2. remove columns with duplicate name 
        data = data.loc[:,~data.columns.duplicated()] 
        # 3. remove NAs
        
        data.dropna(axis=0, how = "all" , inplace= True)
        data.dropna(axis=1, how ="all", inplace= True)

        # remove the row if target column has NA 
        try:
            data.dropna(subset = [self.target],inplace=True)
        except KeyError:
            pass


        if self.display_types:
            self.display_dtypes()
        # 4. drop id columns 
        data.drop(self.id_columns,axis=1 ,errors ="ignore",inplace=True)

        return data

    def transform(self, dataset, y= None):
        """
        args :
            dataset : accept a pandas dataframe 
        returns :
            pandas dataframe
        """ 
        data =dataset.copy()

        # also make sure that all the column names are string 
        data.columns = [str(i) for i in data.columns]

        # drop any columns that were asked to drop 
        data.drop(columns = self.features_to_drop, errors = "ignore", inplace= True)
        data = data[self.training_columns]

        # also make sure that all the column names are string 
        data.columns = [str(i) for i in data.columns]

        # if there are inf or -inf then replace them with NaN 

        data.replace([np.inf, -np.inf], np.NaN, inplace=True)

        try:
            data.dropna(subset = [self.target],inplace=True)
        except KeyError:
            pass

        for col in self.training_columns:
            if col not in data.columns:
                raise TypeError(
                    f"test data does not have column {i} which was used for training."
                )
        # just keep picking the data and keep applying to the test data set (be mindful of target variable)
        for col in data.columns:
            if col == self.target and ((self.ml_usecase == "classification") and (self.learnd_dtypes[self.target] == "object")):
                self.le = LabelEncoder()
                data[col] = self.le.transform(data[col].apply(str).astype("object"))
                data[col] = data[col].astype("int64")
            else:
                if self.learnd_dtypes[col].name == "datetime64[ns]":
                    data[col] = pd.to_datetime(data[col], infer_datetime_format=True, utc= False, errors = "coerce")
                data[col] = data[col].astype(self.learnd_dtypes[col])
        
        data.drop(self.id_columns, axis=1, errors="ignore",inplace=True)

        return data 

    # fit_transform
    def fit_transform(self,dataset, y= None):
        data = dataset 

        # since this is for training, we dont need any transformation since it has already been transformed during fit process
        data = self.fit(data)

        # for ml use case 
        if (self.ml_usecase == "calssification") & (data[self.target].dtype =="object"):
            self.le = LabelEncoder()
            data[self.target] = self.fit_transform(data[self.target].apply(str).astype("object"))
            self.replacement = _get_labelencoder_reverse_dict(self.le)

        # drop id columns 
        data.drop(self.id_columns, axis= 1, errors ="ignore", inplace=True)
        # finally save a list of columns that we would need from test data set 
        self.final_training_columns = data.columns.to_list()
        self.final_training_columns.remove(self.target)

        return data 
        

    def display_dtypes(self):
        display(
            wg.Text(
                value = "Following data types have been inferred automatically, if they are correct press enter to continue or type 'quit' otherwise. ",
                layout = Layout(width="100%")
            ), display_id = "m1"
        )
        dt_print_out = pd.DataFrame(
                self.learnd_dtypes, columns=["Feature_Type"]
            ).drop("UNSUPERVISED_DUMMY_TARGET", errors="ignore")
        dt_print_out["Data Type"] = ''
        for i in dt_print_out.index:
            if i != self.target:
                if i in self.id_columns:
                    dt_print_out.loc[i, "Data Type"] = "ID Column"
                elif dt_print_out.loc[i, "Feature_Type"] == "object":
                    dt_print_out.loc[i, "Data Type"] = "Categorical"
                elif dt_print_out.loc[i, "Feature_Type"] == "float32":
                    dt_print_out.loc[i, "Data Type"] = "Numeric"
                elif dt_print_out.loc[i, "Feature_Type"] == "datetime64[ns]":
                    dt_print_out.loc[i, "Data Type"] = "Date"
                # elif dt_print_out.loc[i,'Feature_Type'] == 'int64':
                #  dt_print_out.loc[i,'Data Type'] = 'Categorical'
            else:
                dt_print_out.loc[i, "Data Type"] = "Label"

            # if we added the dummy  target column , then drop it
        dt_print_out.drop(index="dummy_target", errors="ignore", inplace=True)

        display(dt_print_out[["Data Type"]])
        self.response = input()

        if self.response in [
                "quit",
                "Quit",
                "exit",
                "EXIT",
                "q",
                "Q",
                "e",
                "E",
                "QUIT",
                "Exit",
            ]:
            sys.exit(
                    "Read the documentation of setup to learn how to overwrite data types over the inferred types. setup function must run again before you continue modeling."
                )

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Imputation 

class Simple_Imputer(_BaseImputer):
    """
    Imputes all type of data (numerical, categorical and time).
    -> Highly recommended to run define_Datatypes class first
    Numerical values -> can be imputed with mean/ median or filled with zeros 
    Categorical values ->  can be replaced with "others"/ "most-frequent" or "constant" 
    Time values -> imputed with the most frequent value (***)
    Ignores target(y) variable

    Args: 
        Numeric_strategy : string (one of {"mean","median","zero","most_frequent"})
        categorical_strategy : string, (one of {"not_available", "most_frequent"})
        target: string, name of the target variable
    """

    _numeric_strategies = {
        "mean": "mean",
        "median": "median",
        "most_frequent": "most_frequent",
        "zero": "constant"
    }
    _categorical_strategies = {
        "most_frequent": "most_frequent",
        "not_available" : "constant"
    }

    def __init__(self,numeric_strategy, categorical_strategy,target_variable, fill_value_numerical = 0, fill_value_categorical = "not_available"):
        if numeric_strategy not in self._numeric_strategies:
            numeric_strategy = "zero"
        self.numeric_strategy = numeric_strategy
        self.target = target_variable
        if categorical_strategy not in self._categorical_strategies:
            categorical_strategy = "most_frequent"
        self.categorical_strategy = categorical_strategy
        self.numeric_imputer = SimpleImputer(
            strategy = self._categorical_strategies[self.categorical_strategy],
            fill_value = fill_value_numerical
        )
        self.categorical_imputer = SimpleImputer(
            strategy = self._categorical_strategies[self.categorical_strategy],
            fill_value = fill_value_categorical
        )
        self.most_frequent_time = []



    def fit(self, dataset,y = None):
        try:
            data =dataset.drop(self.target , axis=1)
        except:
            data = dataset 

        self.numeric_columns = data.select_dtypes(include = ["float32","int64"]).columns
        self.categorical_columns = data.select_dtypes(include= ['object']).columns
        self.time_columns = data.select_dtypes(include = ['datetime64[ns]'])

        statistics = []

        if not self.numeric_columns.empty:
            self.numeric_imputer.fit(data[self.numeric_columns])
            statistics.append((self.numeric_imputer.statistics_, self.numeric_columns))
        
        if not self.categorical_columns.empty:
            self.categorical_imputer.fit(data[self.categorical_columns])
            statistics.append((self.categorical_imputer.statistics_, self.categorical_columns))

        if not self.time_columns.empty:
            self.most_frequent_time = []
            for col in self.time_columns:
                self.most_frequent_time.append(data[col].mode()[0])
            statistics.append((self.most_frequent_time, self.time_columns))

        self.statistics_ = np.zeros(shape=len(data.columns), dtype=object)
        columns = list(data.columns)
        for s, index in statistics:
            for i, j in enumerate(index):
                self.statistics_[columns.index(j)] = s[i]

        return self 

    def transform(self, dataset, y=None):
        data = dataset
        imputed_data = []
        if not self.numeric_columns.empty:
            # self.numeric columns가 비어 있지 않다면 
            numeric_data = pd.DataFrame(
                self.numeric_imputer.transform(data[self.numeric_columns]),columns = self.numeric_columns, index= data.index
            )
            imputed_data.append(numeric_data)
        if not self.categorical_columns.empty:
            categorical_data = pd.DataFrame(
                self.categorical_imputer.transform(data[self.categorical_columns]),
                columns=self.categorical_columns,
                index=data.index,
            )
            for col in categorical_data.columns:
                categorical_data[col] = categorical_data[col].apply(str)
            imputed_data.append(categorical_data)

        if not self.time_columns.empty:
            time_data = data[self.time_columns]
            for i, col in enumerate(time_data.columns):
                time_data[col].fillna(self.most_frequent_time[i])
            imputed_data.append(time_data)

        if imputed_data:
            data.update(pd.concat(imputed_data,axis=1))
        data.astype(dataset.dtypes)

        return data

    def fit_transform(self, dataset, y = None):
        data = dataset 
        self.fit(data)
        return self.transform(data)


class Surrogate_Imputer(_BaseImputer):
    """
    
    
    """

# now get the replacement dict
def _get_labelencoder_reverse_dict(le: LabelEncoder) -> dict:
    rev = le.inverse_transform(range(0,len(le.classes_)))
    rep = np.array(range(0, len(le.classes_)))
    replacement = {}
    for i,k in zip(rev,rep):
        replacement[i] = k
    return replacement

# preprocess_all_in_one
def preprocess_all_in_one(
    train_data, target_variable, ml_usecase= None,test_data = None, 
    categorical_features = [], numerical_features =[], time_features =[], 
    features_to_drop =[],
    display_types=True, 
    id_columns = [],
    imputation_type = "simple",
    numeric_imputation_strategy = "mean",
    categorical_imputation_strategy = "not_available"):
    """
    Following preprocess steps are taken: 
    - 1) Auto infer data types (you can designate types) : id_columns가 있으면 직접 넣어줘야 합니다. 
    - 2) Impute (simple or with surrogate columns or Iterative Imputer)

    """
    # also make sure that all the column names are string 
    train_data.columns = [str(i) for i in train_data.columns]
    if test_data is not None: 
        test_data.columns = [str(i) for i in test_data.columns]
    
    if target_variable is None:
        ml_usecase = "regression"
    else:
        # we need to auto infer the ml use case 
        inferred_ml_usecase, subcase = infer_ml_usecase(train_data[target_variable])
        if ml_usecase is None:
            ml_usecase = inferred_ml_usecase
    
    dtypes = DataTypes_Auto_infer(
        target = target_variable, 
        ml_usecase = ml_usecase, 
        categorical_features = categorical_features, 
        numerical_features = numerical_features, 
        time_features= time_features, 
        features_to_drop = features_to_drop, 
        display_types = display_types,
        id_columns = id_columns
    )
    # for imputation 
    # imputation_type = "A"
    if imputation_type == "simple":
        imputer = Simple_Imputer(
            numeric_strategy = numeric_imputation_strategy,
            target_variable = target_variable,
            categorical_strategy = categorical_imputation_strategy
        )
    else: 



    pipe = Pipeline([
        ("dtypes", dtypes),
        ("imputer",imputer)
    ])
    return pipe 