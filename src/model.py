from cgi import test
import json
from pathlib import Path
import joblib
import pandas as pd



def algo(mode):
    X_test=pd.read_csv(Path('Balanced_processed_data\X_test.csv'))
    X_train=pd.read_csv(Path('Balanced_processed_data\X_train.csv'))
    y_test=pd.read_csv(Path('Balanced_processed_data\y_test.csv'))
    y_train=pd.read_csv(Path('Balanced_processed_data\y_train.csv'))
    model=mode
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    print('____________________________________\n',model,'Report\n____________________________________')
    train_score=model.score(X_train,y_train)
    test_score=model.score(X_test,y_test)
    #test_score.to_csv(Path('report\test_score.csv'),index=False)
    joblib.dump(model,Path('model\model_rfc.pkl'))
    path=Path('report\metrics.json')
    with open(path, "w") as f:
        
        scores = {
            "train_score" : train_score,
            "test_score": test_score,
             }
        json.dump(scores,f)
        
    