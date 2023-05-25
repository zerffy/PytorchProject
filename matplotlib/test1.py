import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

data = [1, 2, 3, 4]
s1 = [4, 8, 2, 6]
s2 = [10, 12, 5, 3]
# 设置字体为楷体
mpl.rcParams['font.sans-serif'] = ['KaiTi']
# plt.plot(data, s1)
# plt.plot(data, s2)
# 设置格式
plt.plot(data, s1, "ro--", label="s1")
plt.plot(data, s2, "b^--", label="s2")
# 添加标题
plt.title("图一")
# 标注x，y轴
plt.xlabel('x')
plt.ylabel('y')
# 添加图列
plt.legend()
# 设置坐标刻度
plt.xticks([1, 2, 3, 4])
plt.yticks(np.arange(2, 13, 1))
# 添加辅助线网格
plt.grid()
# 调整显示范围
plt.xlim(2.5, 4.5)
plt.ylim(1.5, 6.5)
plt.show()