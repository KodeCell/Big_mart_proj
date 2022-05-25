import pandas as pd
import numpy as np

def preprocess(train):

    # Since the Range of the output is too high we normalize it using log transformation
    train['Item_Outlet_Sales'] = np.log(1+train['Item_Outlet_Sales'])
    train.columns = train.columns.str.lower()

    # getting the data after dealing with categorical variables
    import feature_eng
    data = feature_eng.encode_cat(train)

    # sacling the train and test data
    Y = data.item_outlet_sales.values
    X = data.drop('item_outlet_sales',axis = 1)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled,Y