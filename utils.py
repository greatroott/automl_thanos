from typing import Tuple
import pandas as pd


def infer_ml_usecase(y:pd.Series) -> Tuple[str,str]:
    c1 = "int" in y.dtype.name
    c2 = y.nunique() <= 20 
    c3 = y.dtype.name in ["object","bool","category"]

    if (c1 and c2) or c3:
        ml_usecase = "classification"
    else:
        ml_usecase = "regression"

    if y.nunique() > 2 and ml_usecase != "regression":
        subcase = "multi"
    else:
        subcase = "binary"

    return ml_usecase, subcase







# def check_metric(actual: pd.Series, prediciton: pd.Series, metric: str, round: int = 4): -> result : float
#     """
#     This is a function to evaluate classification and Regression metrics 

#     actual : pandas.Series 
#         Actual values of the target variable 

#     prediction : pandas.Series
#         Predicted values of the target variable

#     metric : str
#         Metric to use. 

#     round: integer, default = 4
#         number of decimal places the metrics will be rounded to 

#     Returns: 
#         result : float
#         The result of the evaluation  
         
#     """
#     # general 
#     import .containers.metrics.classification

#     globals_dict = {"y" : prediciton}
#     metrics_containers =



if __name__ == "__main__":
    import os 
    os.chdir("../")
    os.chdir("./dataset/titanic")
    