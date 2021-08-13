import numpy as np
import pandas as pd
import time
from sklearn import preprocessing

def get_y(a):
    change = []
    times = []
    result = []
    after_proc = a.lstrip().split(',')
    for i in after_proc:
        j = i.split('：')
        change.append(j[0])
        times.append(j[1].lstrip())
    change.reverse()
    times.reverse()
    # print(times)
    length = len(times)

    st = 0
    st_reverse = 0
    #第一大分类
    if int(times[0][0:4]) > 2020:
        if (change[0][0] == 'S') or (change[0][0] == '*') or (change[0][0] == '股'):
            st_reverse = 0
            st = 1
        elif change[0][0] == '去':
            st_reverse = 1
            st = 0

        for j in range(12):
            result.append(st_reverse)
        result.append(st)

    #第二大分类
    elif int(times[0][0:4]) > 2009:
        for i in range(0, length):
            if i == 0:
                # 将此位置之前的年份进行判断
                gap1 = int(times[i][0: 4]) - 2009
                if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                    st_reverse = 0
                    st = 1
                elif change[i][0] == '去':
                    st_reverse = 1
                    st = 0
                for j in range(0, gap1):
                    result.append(st_reverse)
                # 将此位置后面的年份进行判断
                if length > 1:
                    if (int(times[i][0: 4]) <= 2009) and (int(times[i + 1][0: 4]) >= 2009):
                        gap = int(times[i + 1][0: 4]) - 2009
                        if (length == 2) and int(times[length - 1][0: 4]) == 2021:  gap = gap + 1
                        if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                            st = 1
                        elif change[i][0] == '去':
                            st = 0
                        for j in range(0, gap):
                            result.append(st)
                        if change[i + 1][0] == '去':
                            result[-1]=0
                    elif (int(times[i][0: 4]) <= 2020) and (int(times[i + 1][0: 4]) > 2020):
                        gap = 2020 - int(times[i][0: 4])
                        if(length==2) and int(times[length-1][0: 4])==2021:  gap=gap+1
                        if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                            st = 1
                        elif change[i][0] == '去':
                            st = 0
                        for j in range(0, gap + 1):
                            result.append(st)
                        if change[i + 1][0] == '去':
                            result[-1]=0
                    elif int(times[i][0: 4]) < 2009 or int(times[i][0: 4]) > 2020:
                        pass
                    else:
                        gap = int(times[i + 1][0: 4]) - int(times[i][0: 4])
                        if(length==2) and int(times[length-1][0: 4])==2021:  gap=gap+1
                        if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                            st = 1
                        elif change[i][0] == '去':
                            st = 0
                        for j in range(0, gap):
                            result.append(st)
                        if change[i + 1][0] == '去':
                            result[-1]=0

                else:
                    gap = 2020 - int(times[i][0: 4])
                    for j in range(0, gap+1+1):
                        result.append(st)
            else:
                if i == length-1 and int(times[i][0: 4]) <= 2020:
                    gap = 2020 - int(times[i][0: 4])
                    if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                        st = 1
                    elif change[i][0] == '去':
                        st = 0
                    for j in range(0, gap +1+1):
                        result.append(st)
                elif (int(times[i][0: 4]) <= 2009) and (int(times[i + 1][0: 4]) >= 2009):
                    gap = int(times[i + 1][0: 4]) - 2009
                    if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                        st = 1
                    elif change[i][0] == '去':
                        st = 0
                    for j in range(0, gap):
                        result.append(st)
                    if i == length - 2 and int(times[i + 1][0: 4]) > 2020:
                        result.append(1)
                elif (int(times[i][0: 4]) <= 2020) and (int(times[i + 1][0: 4]) > 2020):
                    gap = 2020 - int(times[i][0: 4])
                    if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                        st = 1
                        st_reverse = 1
                    elif change[i][0] == '去':
                        st = 0
                        st_reverse = 1
                    for j in range(0, gap + 1):
                        result.append(st)
                    result.append(st_reverse)

                elif int(times[i][0: 4]) < 2009 or int(times[i][0: 4]) > 2020:
                    pass
                else:
                    gap = int(times[i + 1][0: 4]) - int(times[i][0: 4])
                    if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                        st = 1
                    elif change[i][0] == '去':
                        st = 0

                    if(i==length-1):
                        for j in range(0, gap+1):
                            result.append(st)
                    else:
                        for j in range(0, gap):
                            result.append(st)

    else:
        for i in range(0, length):
            if i == length - 1 and int(times[i][0: 4]) <= 2020:
                if int(times[i][0: 4]) <= 2009:
                    gap = 2020 - 2009
                else:
                    gap = 2020 - int(times[i][0: 4])
                if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                    st = 1
                elif change[i][0] == '去':
                    st = 0
                for j in range(0, gap+1+1):
                    result.append(st)
            elif (int(times[i][0: 4]) <= 2009) and (int(times[i + 1][0: 4]) >= 2009):
                gap = int(times[i + 1][0: 4]) - 2009
                if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                    st = 1
                elif change[i][0] == '去':
                    st = 0
                for j in range(0, gap):
                    result.append(st)
                if i==length-2 and int(times[i+1][0: 4])>2020:
                    result.append(1)
            elif (int(times[i][0: 4]) <= 2020) and (int(times[i + 1][0: 4]) > 2020):
                gap = 2020 - int(times[i][0: 4])
                if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                    st = 1
                    st_reverse=1
                elif change[i][0] == '去':
                    st = 0
                    st_reverse=1
                for j in range(0, gap + 1):
                    result.append(st)

                result.append(st_reverse)
            elif int(times[i][0: 4]) < 2009 or int(times[i][0: 4]) > 2020:
                pass
            else:
                gap = int(times[i + 1][0: 4]) - int(times[i][0: 4])
                if (change[i][0] == 'S') or (change[i][0] == '*') or (change[i][0] == '股'):
                    st = 1
                elif change[i][0] == '去':
                    st = 0
                if (i == length - 1):
                    for j in range(0, gap + 1):
                        result.append(st)
                else:
                    for j in range(0, gap):
                        result.append(st)
    #如果当年ST那么不可能立马出现问题，而是一个渐变过程
    for i in range(2,len(result)):
        if result[i]==1:
            result[i-1]=1
            #result[i-2]=1

    return result

dmzm = pd.read_csv("/Users/jackieshi/PycharmProjects/pythonProject4/全部股票戴帽摘帽.csv", encoding='gbk', index_col=0)
data_dmzm = pd.read_csv('/Users/jackieshi/PycharmProjects/pythonProject4/全部股票戴帽摘帽.csv', encoding='gbk',
                        index_col=0).iloc[0:4313, -1]
data_dmzm=data_dmzm.dropna(axis = 0)
dmzm=dmzm.dropna(axis=0)
print(dmzm)

name=[]
new=[]
count = 0
for i in range(0, len(dmzm.index.to_list())):
    name.append(data_dmzm.index[i])
    result = get_y(data_dmzm[i])
    print(result)
    new.append(result)
    count += 1
    #print(count)
    #if len(result)!=13:
        #print(count)
pd.concat([pd.DataFrame(name),pd.DataFrame(new)],axis=1).to_csv("DMZM_data/全部股票戴帽摘帽处理后V4.csv")


# y_train = get_y(a)
# print(y_train)
