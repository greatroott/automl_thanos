from preprocess import * 
import os 
data_path = os.path.abspath("../dataset")
import pandas as pd 

# titanic-test 
for i in ['titanic','costarican','home-credit','porto','walmart']:
    print(f"---- {i} test 시작 ----")
    if i == "titanic":
        target_ = "Survived"
    elif i == "walmart":
        target_ = "VisitNumber"
    elif i == "porto":
        target_ = "target"
    elif i == "home-credit":
        target_ = "TARGET"
    elif i == "costarican":
        target_ = "Target"
    else:
        raise KeyError

    df = pd.read_csv(os.path.join(data_path,f"{i}/train.csv"))
    test = pd.read_csv(os.path.join(data_path,f"{i}/test.csv"))
    df.iloc[:,(df.isnull().sum()!=0).values].to_csv(f"./origin_{i}.csv",index=False)
    pipe = preprocess_all_in_one(train_data = df, target_variable = target_, ml_usecase = "classification",test_data =test ,display_types= False,imputation_type = "simple")
    check = pd.read_csv(f"./origin_{i}.csv")
    check_cols = check.columns
    fit_transform_test = pipe.fit_transform(df)
    
    check = pd.concat([check, fit_transform_test[check_cols]],axis=1)
    check.to_csv(f"./origin_{i}.csv",index=False)

    print(f"{i}-test_set_Apply no error")

    print()



