import numpy as np
import pandas as pd
import time
from sklearn import preprocessing
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 设置最大显示列和行数
# pd.set_option('display.max_columns', 50)
# pd.set_option('display.max_rows', 50)
# 公司随时间变化的st数据
st_data = pd.read_csv("/Users/jackieshi/PycharmProjects/pythonProject4/2021ST公司带帽摘帽处理后.csv",
                      encoding="GBK", index_col=0)
# 总的公司指标数据
data = pd.read_csv("/Users/jackieshi/PycharmProjects/pythonProject4/风险警示.csv",
                   encoding="GBK", index_col=0)
stock_list_init = st_data.iloc[:, 0].to_list()
print(data)
print(len(stock_list_init))

stock_list = []
Size = []
X_train = []
y_train = []
X_test = []
y_test = []

for i in range(0, len(stock_list_init)):
    try:
        stock_df = pd.read_csv("ST_dataV5/" + stock_list_init[i].replace(".", "_") + ".csv", index_col=0).iloc[0:12,
                   0:42]
        # print("=============")
        # print(stock_df)
        # print("=============")
    except BaseException as e:
        continue
    for m in range(0, 12):
        for n in range(0, 42):
            stock_df.iloc[m, n] = np.float64(str(stock_df.iloc[m, n]).replace(',', ''))
    # 用相邻后面（back）特征填充前面空值
    stock_df.fillna(method='bfill', inplace=True)
    # 用相邻前面（front）特征填充前面空值
    stock_df.fillna(method='ffill', inplace=True)
    stock_df = stock_df.fillna(0)
    stock_list.append(stock_list_init[i].replace(".", "_"))
    # 拿该股票对应的st标签
    result = st_data.iloc[i, 1:].to_list()
    print(result)
    print('===============')
    print(stock_df)
    for k in range(4, 12):
        data2 = np.array(preprocessing.minmax_scale(stock_df, feature_range=(0, 1), axis=0))[k - 4:k + 1, :]
        data2 = pd.concat([pd.DataFrame(data2), pd.DataFrame(result[k - 4:k + 1])], axis=1)
        if k <= 8:
            X_train.append(data2)
            # lebel是前5年后的一年结果
            y_train.append(result[k + 1])

        if k == 9 or k == 10:
            X_test.append(data2)
            y_test.append(result[k + 1])

print(len(X_train))
Size.append([np.array(X_train),
             np.array(y_train).reshape(-1, 1),
             np.array(X_test),
             np.array(y_test).reshape(-1, 1)])
# print(Size)
Size = np.array(Size, dtype=object)
print(Size[0][0].shape,
      Size[0][2].shape)
# print(Size[0, 1].shape)
np.save(file="V5_data/V5_data.npy", arr=Size)

