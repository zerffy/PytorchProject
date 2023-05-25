import pandas as pd
pd.set_option('display.unicode.east_asian_width', True)  # 规整 格式
#导入excel
# df = pd.read_excel('data/excel/2021nba.xls')
# df2 = pd.read_excel('data/excel/2021nba.xls', usecols=[0, 1])
# print(df2)
#导入csv
# df = pd.read_csv('data/csv/2021nba.csv', sep=',', encoding='gbk')
# print(df.head())
#导入txt
# df = pd.read_csv('data/txt/设备表1.txt', sep='\t', encoding='gbk', header=None)
# print(df.head())
#导入html
url = 'http://www.espn.com/nba/salaries'
df = pd.DataFrame() #空的
#添加数据
df = df.append(pd.read_html(url, header=0))
print(df)
#保存csv文件
df.to_csv('nbasalary.csv', index=False)