import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute._base import _BaseImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted, check_random_state
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.ensemble import RandomForestRegressor as rfr
from lightgbm import LGBMClassifier as lgbmc
from lightgbm import LGBMRegressor as lgbmr
import numpy as np
from IPython.display import display
import ipywidgets as wg   
from ipywidgets import Layout 
import sys 
from typing import Optional 
import gc 
from _logging import get_logger
from utils import infer_ml_usecase

pd.set_option("display.max_columns",500)
pd.set_option("display.max_rows",500)

SKLEARN_EMPTY_STEP = "passthrough"

def str_if_not_null(x):
    if pd.isnull(x) or (x is None) or pd.isna(x) or (x is not x) : 
        return x 
    return str(x)

# Column Name cleaner transformer
class Clean_Column_Names(BaseEstimator, TransformerMixin):
    '''
    - cleans special chars that are not supported by json format 
    '''
    def fit(self,data,y = None):
        return self 

    def transform(self, dataset,y = None):
        data = dataset 
        data.columns = data.columns.str.replace(r"[\,\}\{\]\[\:\"\']","") 
        return data 

    def fit_transform(self, dataset, y= None):
        return self.transform(dataset,y = y)


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


class Iterative_Imputer(_BaseImputer):
    """
    Multivariate imputer that estimates each feature from all the others. 

    A strategy for imputing missing values by modeling each feature with missing values as a function of other features in a round-robin fashion.

    """
    def __init__(
        self,
        regressor: BaseEstimator,
        classifier: BaseEstimator,
        *,
        target = None,
        missing_values = np.nan, # missing은 np.nan으로 처리
        initial_strategy_numeric: str = "mean",
        initial_strategy_categorical: str = "most_frequent",
        ordinal_columns: Optional[list] = None,
        max_iter: int =10, 
        warm_start: bool = False, 
        imputation_order: str = "ascending",
        verbose: int = 0,
        random_state: int = None,
        add_indicator: bool = False  
        ):
        super().__init__(missing_values = missing_values, add_indicator = add_indicator)
        self.regressor = regressor 
        self.classifier = classifier 
        self.initial_strategy_numeric = initial_strategy_numeric
        self.initial_strategy_categorical = initial_strategy_categorical 
        self.max_iter = max_iter 
        self.warm_start = warm_start 
        self.imputation_order = imputation_order 
        self.verbose = verbose
        self.random_state = random_state
        self.target = target
        if ordinal_columns is None: 
            ordinal_columns = []
        self.ordinal_columns = list(ordinal_columns)
        self._column_cleaner = Clean_Column_Names()

    def _initial_imputation(self,X):
        '''
        Imputer used to initialize the missing values.
        '''
        if self.initial_imputer_ is None:
            self.initial_imputer_ = Simple_Imputer(
                target_variable = "__TARGET__", # dummy value, we don;t actally want to drop anything
                numeric_strategy = self.initial_strategy_numeric,
                categorical_strategy = self.initial_strategy_categorical
            ) 
            X_filled = self.initial_imputer_.fit_transform(X)
        else:
            X_filled = self.initial_imputer_.transform(X)

        
        return X_filled 

    def _impute_one_feature(self, X, column, X_na_mask, fit):
        if not fit:
            check_is_fitted(self)
        is_classification = (
            X[column].dtype.name == "object" or column in self.ordinal_columns
        )
        if is_classification:
            if column in self.classifiers_:
                time, dummy, le, estimator = self.classifiers_[column]
            elif not fit:
                return X
            else:
                estimator = clone(self._classifier)
                time = Make_Time_Features()
                dummy = Dummify(column)
                le = LabelEncoder()
        else:
            if column in self.regressors_:
                time, dummy, le, estimator = self.regressors_[column]
            elif not fit:
                return X
            else:
                estimator = clone(self._regressor)
                time = Make_Time_Features()
                dummy = Dummify(column)
                le = None

        if fit:
            fit_kwargs = {}
            X_train = X[~X_na_mask[column]]
            y_train = X_train[column]
            # catboost handles categoricals itself
            if "catboost" not in str(type(estimator)).lower():
                X_train = time.fit_transform(X_train)
                X_train = dummy.fit_transform(X_train)
                X_train.drop(column, axis=1, inplace=True)
            else:
                X_train.drop(column, axis=1, inplace=True)
                fit_kwargs["cat_features"] = []
                for i, col in enumerate(X_train.columns):
                    if X_train[col].dtype.name == "object":
                        X_train[col] = pd.Categorical(
                            X_train[col], ordered=column in self.ordinal_columns
                        )
                        fit_kwargs["cat_features"].append(i)
                fit_kwargs["cat_features"] = np.array(
                    fit_kwargs["cat_features"], dtype=int
                )
            X_train = self._column_cleaner.fit_transform(X_train)

            if le:
                y_train = le.fit_transform(y_train)

            try:
                assert self.warm_start
                estimator.partial_fit(X_train, y_train)
            except:
                estimator.fit(X_train, y_train, **fit_kwargs)

        X_test = X.drop(column, axis=1)[X_na_mask[column]]
        X_test = time.transform(X_test)
        # catboost handles categoricals itself
        if "catboost" not in str(type(estimator)).lower():
            X_test = dummy.transform(X_test)
        else:
            for col in X_test.select_dtypes("object").columns:
                X_test[col] = pd.Categorical(
                    X_test[col], ordered=column in self.ordinal_columns
                )
        result = estimator.predict(X_test)
        if le:
            result = le.inverse_transform(result)

        if fit:
            if is_classification:
                self.classifiers_[column] = (time, dummy, le, estimator)
            else:
                self.regressors_[column] = (time, dummy, le, estimator)

        if result.dtype.name == "float64":
            result = result.astype("float32")

        X_test[column] = result
        X.update(X_test[column])

        gc.collect()

        return X

    def _impute(self, X, fit: bool):
        if self.target in X.columns:
            target_column = X[self.target]
            X = X.drop(self.target, axis=1)
        else:
            target_column = None

        original_columns = X.columns
        original_index = X.index

        X = X.reset_index(drop=True)
        X = self._column_cleaner.fit_transform(X)

        self.imputation_sequence_ = (
            X.isnull().sum().sort_values(ascending=self.imputation_order == "ascending")
        )
        self.imputation_sequence_ = [
            col
            for col in self.imputation_sequence_[self.imputation_sequence_ > 0].index
            if X[col].dtype.name != "datetime64[ns]"
        ]

        X_na_mask = X.isnull()

        X_imputed = self._initial_imputation(X.copy())

        for i in range(self.max_iter if fit else 1):
            for feature in self.imputation_sequence_:
                get_logger().info(f"Iterative Imputation: {i+1} cycle | {feature}")
                X_imputed = self._impute_one_feature(X_imputed, feature, X_na_mask, fit)

        X_imputed.columns = original_columns
        X_imputed.index = original_index

        if target_column is not None:
            X_imputed[self.target] = target_column
        return X_imputed

    def transform(self, X, y=None, **fit_params):
        return self._impute(X, fit=False)

    def fit_transform(self, X, y=None, **fit_params):
        self.random_state_ = getattr(
            self, "random_state_", check_random_state(self.random_state)
        )
        if self.regressor is None:
            raise ValueError("No regressor provided")
        else:
            self._regressor = clone(self.regressor)
        try:
            self._regressor.set_param(random_state=self.random_state_)
        except:
            pass
        if self.classifier is None:
            raise ValueError("No classifier provided")
        else:
            self._classifier = clone(self.classifier)
        try:
            self._classifier.set_param(random_state=self.random_state_)
        except:
            pass

        self.classifiers_ = {}
        self.regressors_ = {}

        self.initial_imputer_ = None

        return self._impute(X, fit=True)

    def fit(self, X, y=None, **fit_params):
        self.fit_transform(X, y=y, **fit_params)

        return self


# make dummy variables
class Dummify(BaseEstimator, TransformerMixin):
    """
    - makes one hot encoded variables for dummy variable
    - it is HIGHLY recommended to run the Select_Data_Type class first
    - Ignores target variable
      Args: 
        target: string , name of the target variable
  """

    def __init__(self, target):
        self.target = target

        # creat ohe object
        self.ohe = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)

    def fit(self, dataset, y=None):
        data = dataset
        # will only do this if there are categorical variables
        if len(data.select_dtypes(include=("object")).columns) > 0:
            # we need to learn the column names once the training data set is dummify
            # save non categorical data
            self.data_nonc = data.drop(
                self.target, axis=1, errors="ignore"
            ).select_dtypes(exclude=("object"))
            self.target_column = data[[self.target]]
            # # plus we will only take object data types
            categorical_data = data.drop(
                self.target, axis=1, errors="ignore"
            ).select_dtypes(include=("object"))
            # # now fit the trainin column
            self.ohe.fit(categorical_data)
            self.data_columns = self.ohe.get_feature_names(categorical_data.columns)

        return self

    def transform(self, dataset, y=None):
        data = dataset.copy()
        # will only do this if there are categorical variables
        if len(data.select_dtypes(include=("object")).columns) > 0:
            # only for test data
            self.data_nonc = data.drop(
                self.target, axis=1, errors="ignore"
            ).select_dtypes(exclude=("object"))
            # fit without target and only categorical columns
            array = self.ohe.transform(
                data.drop(self.target, axis=1, errors="ignore").select_dtypes(
                    include=("object")
                )
            ).toarray()
            data_dummies = pd.DataFrame(array, columns=self.data_columns)
            data_dummies.index = self.data_nonc.index
            if self.target in data.columns:
                target_column = data[[self.target]]
            else:
                target_column = None
            # now put target , numerical and categorical variables back togather
            data = pd.concat((target_column, self.data_nonc, data_dummies), axis=1)
            del self.data_nonc
            return data
        else:
            return data

    def fit_transform(self, dataset, y=None):
        data = dataset.copy()
        # will only do this if there are categorical variables
        if len(data.select_dtypes(include=("object")).columns) > 0:
            self.fit(data)
            # fit without target and only categorical columns
            array = self.ohe.transform(
                data.drop(self.target, axis=1, errors="ignore").select_dtypes(
                    include=("object")
                )
            ).toarray()
            data_dummies = pd.DataFrame(array, columns=self.data_columns)
            data_dummies.index = self.data_nonc.index
            # now put target , numerical and categorical variables back togather
            data = pd.concat((self.target_column, self.data_nonc, data_dummies), axis=1)
            # remove unwanted attributes
            del (self.target_column, self.data_nonc)
            return data
        else:
            return data


# Time feature extractor
class Make_Time_Features(BaseEstimator, TransformerMixin):
    """
    -Given a time feature , it extracts more features
    - Only accepts / works where feature / data type is datetime64[ns]
    - full list of features is:
      ['month','weekday',is_month_end','is_month_start','hour']
    - all extracted features are defined as string / object
    -it is recommended to run Define_dataTypes first
      Args: 
        time_feature: list of feature names as datetime64[ns] , default empty/none , if empty/None , it will try to pickup dates automatically where data type is datetime64[ns]
        list_of_features: list of required features , default value ['month','weekday','is_month_end','is_month_start','hour']
  """

    def __init__(
        self,
        time_feature=None,
        list_of_features=["month", "weekday", "is_month_end", "is_month_start", "hour"],
    ):
        self.time_feature = time_feature
        self.list_of_features_o = set(list_of_features)

    def fit(self, data, y=None):
        if self.time_feature is None:
            self.time_feature = data.select_dtypes(include=["datetime64[ns]"]).columns
            self.has_hour_ = set()
            for i in self.time_feature:
                if "hour" in self.list_of_features_o:
                    if any(x.hour for x in data[i]):
                        self.has_hour_.add(i)
        return self

    def transform(self, dataset, y=None):
        data = dataset.copy()
        # run fit transform first

        def get_time_features(r):
            features = []
            if "month" in self.list_of_features_o:
                features.append(("_month", str(r.month)))
            if "weekday" in self.list_of_features_o:
                features.append(("_weekday", str(r.weekday())))
            if "is_month_end" in self.list_of_features_o:
                features.append(
                    (
                        "_is_month_end",
                        "1"
                        if calendar.monthrange(r.year, r.month)[1] == r.day
                        else "0",
                    )
                )
            if "is_month_start" in self.list_of_features_o:
                features.append(("_is_month_start", "1" if r.day == 1 else "0"))
            return tuple(features)

        # start making features for every column in the time list
        for i in self.time_feature:
            list_of_features = [get_time_features(r) for r in data[i]]

            fd = defaultdict(list)
            for x in list_of_features:
                for k, v in x:
                    fd[k].append(v)

            for k, v in fd.items():
                data[i + k] = v

            # make hour column if choosen
            if "hour" in self.list_of_features_o and i in self.has_hour_:
                h = [r.hour for r in data[i]]
                data[f"{i}_hour"] = h
                data[f"{i}_hour"] = data[f"{i}_hour"].apply(str)

        # we dont need time columns any more
        data.drop(self.time_feature, axis=1, inplace=True)

        return data

    def fit_transform(self, dataset, y=None):
        # if no columns names are given , then pick datetime columns
        self.fit(dataset, y=y)

        return self.transform(dataset, y=y)



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
    categorical_imputation_strategy = "not_available",
    imputation_classifier = None,
    imputation_regressor = None,
    imputation_max_iter = 10, 
    imputation_warm_start = False,
    imputation_order = "ascending",
    ordinal_columns_and_categories = {}, 
    random_state =42,n_jobs = -1):
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
        # 만약 option에서 이 파트가 없으면 아예 제외해서 집어넣는다~ 
        # 우선 lightgbm으로만 적용
        if imputation_classifier == None:
            imputation_classifier = lgbmc(n_estimators=100, max_depth =5 , n_jobs =n_jobs, random_state = random_state)
        if imputation_regressor == None:
            imputation_regressor = lgbmr(n_estimators=100, max_depth =5, n_jobs= n_jobs, random_state= random_state)
        
        imputer = Iterative_Imputer(
            classifier = imputation_classifier,
            regressor = imputation_regressor, 
            target = target_variable, 
            initial_strategy_numeric = numeric_imputation_strategy, 
            max_iter = imputation_max_iter, 
            warm_start = imputation_warm_start,
            imputation_order = imputation_order, 
            random_state = random_state, 
            ordinal_columns = ordinal_columns_and_categories.keys())



    pipe = Pipeline([
        ("dtypes", dtypes),
        (SKLEARN_EMPTY_STEP,imputer)
    ])

    return pipe 