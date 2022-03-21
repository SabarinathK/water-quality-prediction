from pathlib import Path
import pandas as pd
from collections import Counter
from imblearn.over_sampling import RandomOverSampler 
from sklearn.model_selection import train_test_split


def data_balance():
    X=pd.read_csv(Path('processed_data\processed_label.csv'))
    y=pd.read_csv(Path('processed_data\processed_class.csv'))
    ros = RandomOverSampler()
    X_res, y_res = ros.fit_resample(X, y)
    X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,train_size=0.7,random_state=356)
    X_train.to_csv(Path('Balanced_processed_data\X_train.csv'),index=False)
    X_test.to_csv(Path('Balanced_processed_data\X_test.csv'),index=False)
    y_train.to_csv(Path('Balanced_processed_data\y_train.csv'),index=False)
    y_test.to_csv(Path('Balanced_processed_data\y_test.csv'),index=False)
    
