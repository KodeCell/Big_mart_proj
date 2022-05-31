import pandas as pd


def encode_cat(data):
    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder
    # label encoding
    le = LabelEncoder()
    data['Outlet'] = le.fit_transform(data['outlet_identifier'])  # creating a new column
    label_col =  ['outlet_size','item_fat_content','outlet_location_type','item_type','outlet_type']
    for col in label_col:
        data[col] = le.fit_transform(data[col])

    data.drop(columns= ['item_identifier','outlet_identifier','outlet_establishment_year','new_item_type'],inplace = True)
    return data