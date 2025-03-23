import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def eda(df):
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
    df['Outlet_Size'] = df['Outlet_Size'].fillna('Missing')
    return df

def encode(df):
    oe = OrdinalEncoder()
    df['Item_Fat_Content'] = oe.fit_transform(df[['Item_Fat_Content']])
    df['Outlet_Size'] = oe.fit_transform(df[['Outlet_Size']])
    df['Outlet_Location_Type'] = oe.fit_transform(df[['Outlet_Location_Type']])
    df['Item_Type'] = oe.fit_transform(df[['Item_Type']])
    df['Outlet_Type'] = oe.fit_transform(df[['Outlet_Type']])
    df['Outlet_Establishment_Year'] = oe.fit_transform(df[['Outlet_Establishment_Year']])
    return df

# Function to train the model
def train(df):

    df = eda(df)

    # Encode the categorical columns
    df = encode(df)

    # Train a Random Forest model
    X = df.drop(columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
    y = df['Item_Outlet_Sales']

    rf = GradientBoostingRegressor(n_estimators=300,
                                 learning_rate=0.01,
                                 random_state=100,
                                 max_features= 5
    )
    rf.fit(X, y)
    return rf

def predict_test(df, model):
    # Fill missing values
    ## EDA

    ### 1. Mean imputation for column 'Item_Weight'
    ### 2. Replace NaN with "Missing" in column 'Outlet_Size'

    df = eda(df)
    df = encode(df)

    # Make predictions
    X = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])
    y_pred = model.predict(X)
    return y_pred

if __name__ == '__main__':
    # Load the data
    df = pd.read_csv("train_v9rqX0R.csv")
    df_test = pd.read_csv("test_AbJTz2l.csv")

    model = train(df)
    y_pred = predict_test(df_test, model)
    df_test['Item_Outlet_Sales'] = y_pred
    df_test.to_csv("submission_gbm.csv", columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'], index=False)







