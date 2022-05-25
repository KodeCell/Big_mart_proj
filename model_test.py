from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd


def train(models_list,x_train,x_test,y_train,y_test):
    models = []
    cv_scores = []
    mses = []
    score = []
    for model in models_list:
        models.append(model)
        # training the model
        model.fit(x_train,y_train)
        cv_score = cross_val_score(model, x_train,y_train,cv=5)
        cv_score = np.mean(cv_score)
        cv_scores.append(cv_score)
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test,y_pred)
        mses.append(mse)
        r2score = r2_score(y_test,y_pred)
        score.append(r2score)
    df = pd.DataFrame({'Model': models, 'CV_Score': cv_scores, 'MSE': mses,'R2_Score': score})
    return df
