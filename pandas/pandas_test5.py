# 异常值的监测与处理
import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)  # 规整 格式

df = pd.read_excel('../data/excel/2021nba.xls' ,usecols=['球队', '胜', '排名'])
print(df)
print('\n**********************')
# 改变错误数据
# for x in df.index:
#     if df.loc[x, '排名'] >= 30:
#         df.loc[x, '排名'] = 30
# print(df)
# 将错误数据删除
for x in df.index:
    if df.loc[x, '排名'] >= 30:
        df.drop(x, axis=0, inplace=True)

df = df.sort_values('胜')
print(df)