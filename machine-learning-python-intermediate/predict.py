import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

stocks = pd.read_csv("sphist.csv", parse_dates=["Date"])
stocks = stocks.sort_values(by=["Date"])

# creating new columns
stocks["days_5"] = stocks["Close"].rolling(window=5).mean().shift(1)
stocks["days_30"] = stocks["Close"].rolling(window=30).mean().shift(1)
stocks["std_30"] = stocks["Close"].rolling(window=30).std().shift(1)
stocks["days_365"] = stocks["Close"].rolling(window=365).mean().shift(1)
stocks["ratio_5_365"] = stocks["days_5"] / stocks["days_365"]
stocks["std_5"] = stocks["Close"].rolling(window=5).std().shift(1)
stocks["std_365"] = stocks["Close"].rolling(window=365).std().shift(1)
stocks["std_ratio_5_365"] = stocks["std_5"]/stocks["std_365"]

#volume features
stocks["volume_5"] = stocks["Volume"].rolling(window=5).mean().shift(1)
stocks["volume_365"] = stocks["Volume"].rolling(window=365).mean().shift(1)
stocks["vol_ratio_5_365"] = stocks["volume_5"] /stocks["volume_365"]
stocks["std_vol_5"] = stocks["volume_5"].rolling(window=5).std().shift(1)
stocks["std_vol_365"] = stocks["volume_365"].rolling(window=365).std().shift(1)
stocks["std_vol_ratio_5_365"] = stocks["std_vol_5"]/stocks["std_vol_365"]


#yearly min
stocks["min_365"] = stocks["Close"].rolling(window=365).min().shift(1)
#yearly max
stocks["max_365"] = stocks["Close"].rolling(window=365).max().shift(1)
#ratio current price and min_365
stocks["ratio_to_min_365"] = stocks["Close"] /stocks["min_365"]
#ratio current price and max_365
stocks["ratio_to_max_365"] = stocks["Close"] /stocks["max_365"]
data = stocks.copy()
#dropping null values
data = data[data["Date"] > datetime(year=1951, month=1, day=2)]
data = data.dropna(axis= 0)
numeric_df = data.iloc[:, 7:]
numeric_cols = list(numeric_df.columns)
scaler = MinMaxScaler()
scaler.fit(numeric_df)
numeric_df[numeric_cols] = scaler.fit_transform(numeric_df[numeric_cols])
data_clean = pd.concat([numeric_df, data[["Date", "Close"]]], axis = 1)
data_clean = data_clean.set_index("Date")

#splitting data
date_divider = datetime(year=2013, month=1, day=1)
train = data_clean[:date_divider].copy()
test = data_clean[date_divider: ].copy()
print(train.shape)
print(test.shape)



def train_and_test(df_train, df_test):
    y_train = train["Close"].copy()
    y_test = test["Close"].copy()
    X_train = df_train.drop(["Close"], axis=1)
    X_test = df_test.drop(["Close"], axis=1)
    lr= LinearRegression()
    lr.fit(X_train, y_train)
    prediction = lr.predict(X_test)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    print("mse: {}".format(mse))
    print("rmse: {}".format(rmse))
    return mse, rmse
#trainig and testing
train_and_test(train, test)












