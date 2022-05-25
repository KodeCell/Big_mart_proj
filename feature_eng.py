import pandas as pd


def encode_cat(data):
    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder
    # label encoding
    le = LabelEncoder()
    data['Outlet'] = le.fit_transform(data['outlet_identifier'])  # creating a new column
    label_col = ['outlet_size', 'outlet_location_type', 'outlet_type']
    for col in label_col:
        data[col] = le.fit_transform(data[col])

    # One Hot Encoding
    dummy_col = ['item_fat_content', 'item_type', 'new_item_type']
    data = pd.get_dummies(data, columns=dummy_col, drop_first=True)

    data.drop(columns=['item_identifier', 'outlet_identifier', 'outlet_establishment_year'], inplace=True)

    return data