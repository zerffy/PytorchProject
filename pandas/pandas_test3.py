# 数据抽取
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)  # 规整 格式
data = [[1, 2, 3], [2, 3, 2], [3, 3, 3]]
index = ['湖人', '勇士', '掘金']
columns = ['win', 'lose', 'rate']
df = pd.DataFrame(data=data, index=index, columns=columns)
print(df, "\n****************************")
# 提取行数据
# print(df.loc['湖人'])
# print(df.iloc[0])
# print(df.loc[['湖人', '勇士']], "\n****************************")
# print(df.iloc[[0, 1]], "\n****************************")
# print(df.loc['湖人': '掘金'], "\n****************************")  # 含头含尾
# print(df.iloc[0: 2], "\n****************************")  # 含头不含尾
# print(df.iloc[0::], "\n****************************")

# 提取列数据
# print(df['win'], "\n****************************")
# print(df[['win', 'lose']], "\n****************************")
# 提取某行某列
# 逗号左行右列
# print(df.loc[:, ['win', 'lose']], "\n****************************")
# print(df.iloc[:, [0, 1]], "\n****************************")
# print(df.loc[:, 'win':'lose'], "\n****************************")
# print(df.iloc[:, 0:1], "\n****************************")

# 提取区域数据
# print(df.loc['湖人', 'win'], "\n****************************")
# print(df.loc[['湖人', '掘金'], ['win', 'rate']])
# print(df.iloc[0:2, 0:2])

# 提取指定条件的数据
# print(df.loc[(df['win'] >= 2) & (df['lose'] <=3)])


# 数据的增加、修改和删除
# 按列增加
# df['h'] = [10, 10, 20]
# print(df)

# df.loc[:, 'h'] = [10, 10, 20]
# print(df)
#
# lst = [10, 20, 30]
# df.insert(1, 'l', lst)
# print(df)

# 按行增加
# df.loc['太阳'] = [1, 2, 3]
# print(df)

# new_df = pd.DataFrame(
#     data={'win': [2, 3], 'lose': [1, 2], 'rate': [1, 2]},
#     index=['太阳', '76人']
# )
# print(new_df)
# df = df.append(new_df)
# print(df)

# 修改列标题
# df.columns = ['win', 'lose', 'new']
# print(df)

# df.rename(columns={'win': 'new1', 'lose': 'new2'}, inplace=True)
# print(df)


# 修改行标题
# (1)直接赋值
# df.index = [1, 2, 3]
# # df.index = list('123')
# print(df)
#
# # (2)rename
# df.rename({1: '一', 2: '二'}, inplace=True, axis=0)  # 0修改行，1修改列
# print(df)

# df.loc['湖人'] = [4, 2, 3]
# print(df)
# df.loc[0:2, 'win'] = [7, 8]
# df.iloc[0:2, 0] = [7, 8]
# print(df)


# 删除数据
# 按列删除
# df.drop(['win'], axis=1, inplace=True)
# print(df)

# df.drop(columns='win', inplace=True)
# print(df)

# df.drop(labels='win', axis=1, inplace=True)
# print(df)


# 按行删除
# df.drop(['湖人'],axis=0,inplace=True)
# print(df)

# 带条件的删除
# df.drop(df[df['win'].index[1] < 3], inplace=True)
# print(df)