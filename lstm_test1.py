# import tushare as ts #导入
# cons = ts.get_apis() #建立连接
# #获取沪深指数(000300)的信息，包括交易日期（datetime）、开盘价(open)、收盘价(close)，
# #最高价(high)、最低价(low)、成交量(vol)、成交金额(amount)、涨跌幅(p_change)
# df = ts.bar('000300', conn=cons, asset='INDEX', start_date='2010-01-01', end_date='')
# #删除有null值的行
# df = df.dropna()
# #把df保存到当前目录下的sh300.csv文件中，以便后续使用
# df.to_csv('sh300.csv')
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
from pandas.plotting import register_matplotlib_converters
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# pd.set_option('display.unicode.east_asian_width', True)  # 规整 格式
# df = pd.read_csv('data/csv/sh300.csv', sep=',', encoding='gbk')
# # print(df.head())
# print('\n**********************')
# print(df.describe())

n = 30
LR = 0.001
EPOCH = 100
train_end = -500

def readData(column='high', n=30, all_too=True, index=False, train_end=-500):
    df = pd.read_csv("data/csv/sh300.csv", index_col=0)
    # 以日期为索引
    df.index = list(map(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"), df.index))
    # 获取每天的最高价
    df_column = df[column].copy()
    # 拆分为训练集和测试集
    df_column_train, df_column_test = df_column[:train_end], df_column[train_end - n:]
    # 生成训练数据
    df_generate_from_df_column_train = generate_df_affect_by_n_days(df_column_train, n, index=index)
    if all_too:
        return df_generate_from_df_column_train, df_column, df.index.tolist()
    return df_generate_from_df_column_train
# 其中最后一列维标签数据。就是把当天的前n天作为参数，当天的数据作为label
def generate_df_affect_by_n_days(series, n, index=False):
    if len(series) <= n:
        raise Exception("The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = pd.DataFrame()
    for i in range(n):
        df['c%d' % i] = series.tolist()[i:-(n - i)]
    df['y'] = series.tolist()[n:]
    if index:
        df.index = series.index[n:]
    return df


register_matplotlib_converters()

class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out)
        return out


class TrainSet(Dataset):
    def __init__(self, data):

        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)

# 数据集建立 #对数据进行预处理，规范化及转换为Tensor
df, df_all, df_index = readData('high', n=n, train_end=train_end)

df_all = np.array(df_all.tolist())
df_numpy = np.array(df)

df_numpy_mean = np.mean(df_numpy)
df_numpy_std = np.std(df_numpy)

df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(df_numpy)

trainset = TrainSet(df_tensor)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)


device = torch.device("cuda")
rnn = RNN(n)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

for step in range(EPOCH):
    for tx, ty in trainloader:
        output = rnn(torch.unsqueeze(tx, dim=0))
        loss = loss_func(torch.squeeze(output), ty)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(step, loss)
    if step % 10:
        torch.save(rnn, 'rnn.pkl')
torch.save(rnn, 'rnn.pkl')

generate_data_train = []
generate_data_test = []

test_index = len(df_all) + train_end

df_all_normal = (df_all - df_numpy_mean) / df_numpy_std
df_all_normal_tensor = torch.Tensor(df_all_normal)
for i in range(n, len(df_all)):
    x = df_all_normal_tensor[i - n:i]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)
    y = rnn(x)
    if i < test_index:
        generate_data_train.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)
    else:
        generate_data_test.append(torch.squeeze(y).detach().numpy() * df_numpy_std + df_numpy_mean)

# plt.plot(df_index, df_all, label='原始数据')
# # plt.plot(df_index[n:train_end], generate_data_train, label='训练数据')
# plt.plot(df_index[train_end:], generate_data_test, label='测试数据')

# plt.plot(df_index[train_end:-400], df_all[train_end:-400], label='原始数据')
# plt.plot(df_index[train_end:-400], generate_data_test[:-400], label='测试数据')

plt.plot(df_index[500:700], df_all[500:700], label='原始数据')
plt.plot(df_index[500:700], generate_data_train[500:700], label='训练数据')

plt.legend()
plt.show()
