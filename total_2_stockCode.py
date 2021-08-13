

# 把总的数据（1个excel）分为公司代码
# 输出的是各公司09～20年所有指标

import pandas as pd
from pandas import DataFrame
import numpy as np
frame_list =["风险警示"]#["信息技术","公共事业","可选消费","工业","房地产","日常消费","材料","电信服务","能源"]
def framepad(framename):
    print('framepad process is:\n', framename)
    print('36 colume:\n', framename.iloc[:, [36]])
    for i in range(0, len(framename.iloc[0,:])):
        if(i==40 or i==36):
            framename.iloc[:, i].fillna(method='pad', inplace=True)
            framename.iloc[:, i].fillna(0, inplace=True)
        else:
            framename.iloc[:, i].fillna(method='pad', inplace=True)
            framename.iloc[:, i].fillna(method='ffill', inplace=True)
    return framename


def split_frame_ST(framename):
    for i in range(0,len(framename)):
        try:
            print(framename.iloc[i, 0])
            # get the code of the stock
            code_name = framename.iloc[i, 0].replace(".", "_")
            new_frame = pd.DataFrame(np.array(framename.iloc[i, 2:]).reshape(-1, 12).T)
            frame = framepad(new_frame)
            new=pd.DataFrame(framename.iloc[i, -1:])
            frame=pd.concat([frame,new],axis=1)
            frame.to_csv("ST_dataV5/" + code_name + ".csv", encoding='utf-8-sig')
        except AssertionError as e:
            continue

def run():
    for i in range(0, len(frame_list)):
        data = pd.read_excel("/Users/jackieshi/PycharmProjects/pythonProject4/原始数据/" + frame_list[i] + ".xlsx", engine='openpyxl')
        print('this is data:\n', data)
        split_frame_ST(data)
run()