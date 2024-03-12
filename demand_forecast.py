import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch import Tensor

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# df = pd.read_csv("data/demand.csv", index_col="ID")
df = pd.read_csv("data/demand.csv")
df = df[["ID", "Store ID", "Units Sold"]]
df = df[:100]

df = df.astype(float)

target_sensor = "Units Sold"
feature_cols = list(df.columns.difference([target_sensor]))

forecast_lead = 1
target = f"{target_sensor}_lead_{forecast_lead}"

lt = []
for x, y_df in df.groupby(feature_cols):
    y_df[target] = y_df[target_sensor].shift(-forecast_lead, fill_value=y_df[target_sensor].mean())
    lt.append(y_df)

df2 = pd.concat(lt)


def create_dataset_df(dataset, lookback):
    """Transform a time series into a prediction dataset

    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for z, z_df in dataset.iterrows():
        _feature, _target = z_df[feature_cols + [target_sensor]].values, z_df[[target]].values
        X.append(_feature)
        y.append(_target)
    return torch.tensor(X).to(device), torch.tensor(y).to(device)


test_start = df2.shape[0] // 2

df_train = df2.iloc[:test_start].copy()
df_test = df2.iloc[test_start:].copy()

print("Test set fraction:", len(df_test) / len(df))

lookback = 1
X_train, y_train = create_dataset_df(df_train, lookback=lookback)
X_test, y_test = create_dataset_df(df_test, lookback=lookback)


class OGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)

    def forward(self, x: Tensor):
        # x.to()
        # x = x.to(torch.float32)
        # print(x)
        x, _ = self.lstm(x)
        # x = F.relu(x)
        x = self.linear(x)
        # print(x)
        return x


learning_rate = 5e-5

model = OGModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=2)

n_epochs = 11
for epoch in range(n_epochs):
    model.train().to(device)
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(torch.float32)
        y_batch = y_batch.to(torch.float32)
        # print(X_batch)
        y_pred = model(X_batch).to(device)
        loss = loss_fn(y_pred, y_batch).to(device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Validation
    if epoch % 10 != 0:
        continue
    model.eval()
    with torch.no_grad():
        y_pred = model(X_train.to(torch.float32)).to(device)
        train_rmse = np.sqrt(loss_fn(y_pred, y_train)).to(device)
        y_pred = model(X_test.to(torch.float32)).to(device)
        test_rmse = np.sqrt(loss_fn(y_pred, y_test)).to(device)
    print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))


df[feature_cols].plot.bar()
plt.show()

# with torch.no_grad():
#     # shift train predictions for plotting
#     train_plot = np.ones_like(timeseries) * np.nan
#     y_pred = model(X_train)
#     # print(y_pred)
#     y_pred = y_pred[:, -1, :]
#     # print(y_pred)
#     train_plot[lookback:train_size] = model(X_train)[:, -1, :]
#     # shift test predictions for plotting
#     test_plot = np.ones_like(timeseries) * np.nan
#     test_plot[train_size+lookback:len(timeseries)] = model(X_test)[:, -1, :]
# # plot
# plt.plot(timeseries, c='b')
# plt.plot(train_plot, c='r')
# plt.plot(test_plot, c='g')
# plt.show()