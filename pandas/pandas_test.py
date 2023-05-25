import pandas as pd
# code1
# df = pd.read_excel('data/excel/2021nba.xls')
# print(df)

# code2
# data1 = ['john', 'li', 'yang']
# s = pd.Series(data=data1, index=[1, 2, 3])
# # print(s)
# # print(type(s))
# # print(s[2])
# # print(s[[1, 2]])
# print(s[0:2])  #含头不含尾

# code3
# data1 = ['john', 'li', 'yang']
# s = pd.Series(data=data1, index=['n1', 'n2', 'n3'])
# # print(s)
# # print(s["n1":"n3"])  #含头含尾
# print(s.index)
# print(list(s.index))
# print(s.values)

# code4
# data = [['湖人', 1.1, 1], ['勇士', 2.2, 2], ['掘金', 3.3, 3]]
# columns = ['team', 'sor', 'rate']
# df = pd.DataFrame(data, columns=columns)
# print(df)
#
# data2 = {'team': ['凯尔特人', '热火', '76人'],
#          'sor': [1.1, 2.2 , 3.3],
#          'rate': [1, 2, 3],
#          'dong': 'dong'}
# df2 = pd.DataFrame(data2)
# print(df2)

# code5
# data = {'team': ['凯尔特人', '热火', '76人'],
#          'sor': [1.1, 2.2 , 3.3],
#          'rate': [1, 2, 3]}
# df = pd.DataFrame(data)
# print('值\n', df.values)
# print('类型\n', df.dtypes)
# print('行\n', list(df.index))
# df.index = [1, 2, 3]
# print(df)
# print('行\n', list(df.columns))
# df.columns = ['t', 's ', 'r']
# print(df)
# #行列转换
# pd.set_option('display.unicode.east_asian_width', True)  # 规整 格式
# newdf = df.T
# print(newdf)
#
# print('前n个数据\n', df.head(1))
# print('后n个数据\n', df.tail(1))
#
# print('row:', df.shape[0], 'col:', df.shape[1])
# #查看索引 数据类型 内存信息
# print(df.info)


#code6
data = {'team': ['凯尔特人', '热火', '76人'],
         'sor': [1.1, 2.2, 3.3],
         'rate': [1, 2, 3]}
df = pd.DataFrame(data)
print(df.describe(), '\n**********************')
print(df.count(), '\n**********************')
print(df.sum(), '\n**********************')
print(df.max(), '\n**********************')
print(df.min(), '\n**********************')