# 数据清洗
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)  # 规整 格式

df = pd.read_csv('../data/txt/设备表1.txt', sep='\t', encoding='gbk', header=None)
print(df)
print('\n**********************')
# print(df.info())      # 查看是否有缺失值
# print('\n**********************')
# print(df.isnull())     # 判断缺失值
# print('\n**********************')
# print(df.notnull())

# 处理方法
# 删除
# df = df.dropna()
# 填充
# df[2] = df[2].fillna(0)  # 填充成0
# 删除重复值
# 判断是否有重复值
print(df.duplicated())
# 删除全部重复值
# df = df.drop_duplicates()
# 删除指定列的重复值
# df = df.drop_duplicates([1])
# 保留后一个
# df = df.drop_duplicates([1], keep='last')
# 直接删除，保留一个副本
df1 = df.drop_duplicates([1], inplace=False)
print(df1)
print(df)