import pandas as pd
import numpy as np
train = pd.read_csv('big_mart_train.csv')
test = pd.read_csv('big_mart_test.csv')

# merging the train and test so that we can get more insights of the data
target = train['Item_Outlet_Sales']
train1 = train.drop('Item_Outlet_Sales',axis = 1)
df = pd.concat([train1,test])

df1 = df.copy()

# filling the nan values
df1['Item_Weight'].fillna(df['Item_Weight'].mean(),inplace = True)
# mode[0] as the values of mode can be more than 1
df1['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0],inplace = True)

# converting multiple same values into one
# array(['Low Fat', 'Regular', 'low fat', 'LF', 'reg'], dtype=object)
df1['Item_Fat_Content'] = df1['Item_Fat_Content'].apply(lambda x : x.upper())
df1['Item_Fat_Content'] = df1['Item_Fat_Content'].replace(['LF','REG'],['LOW FAT', 'REGULAR'])
# array(['LOW FAT', 'REGULAR'], dtype=object)

# creating a new attribute
df1['New_Item_Type'] = df1['Item_Identifier'].apply(lambda x : x[:2])

# mapping the new attribute column
df1['New_Item_Type'] = df1['New_Item_Type'].map({'FD' : 'Food','NC' : 'Non-Consumables','DR' : 'Drinks'})

df1.loc[df1['New_Item_Type']=='Non-Consumables', 'Item_Fat_Content'] = 'Non-Edible'

# create small values for establishment year to reduce the higher numbers
df['Outlet_Years'] = 2013 - df['Outlet_Establishment_Year']

df1.to_csv("data_cleaned.csv",index = False)
target.to_csv('train_target.csv',index = False)



