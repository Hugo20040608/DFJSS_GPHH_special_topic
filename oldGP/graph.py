import matplotlib.pyplot as plt

# 資料
categories = ['GP', 'PT', 'RTO', 'RT']
values = [542.09843166, 854.8665291270981, 715.0910492656877, 841.2995323453855]

# 製作長條圖
plt.bar(categories, values)

# 加標題與軸標籤
plt.title('Utilization=0.8')
plt.xlabel('Strategy')
plt.ylabel('mean-flowtime')

# 顯示圖表
plt.savefig('08mean-flowtime.png')


# PT
# Mean mean-flowtime: 854.8665291270981
# Mean makespan: 2714.981397011855
# Mean max-flowtime: 1969.6430836420823

# RTO
# Mean mean-flowtime: 715.0910492656877
# Mean makespan: 2296.2495565411045
# Mean max-flowtime: 1630.223519082333

# RT
# Mean mean-flowtime: 841.2995323453855
# Mean makespan: 2581.907940673282
# Mean max-flowtime: 1988.2708657384965