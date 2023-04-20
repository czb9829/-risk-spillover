# -*- coding: utf-8 -*-
"""
Created on Sat Apr 15 14:40:16 2023

@author: zh
"""
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as stm
import numpy as np
import pandas as pd
import xlrd
import warnings
import math
warnings.filterwarnings('ignore')
import matplotlib

def VARFitter(Data: pd.DataFrame, lags):
    VARModel = stm.tsa.VAR(Data)
    result = VARModel.fit(lags)
    return VARModel, result

def get_Ah(phi,p,h): 
    Ah=np.zeros((31,31))
    if h>=1:
        for i in range(p):
            Ah+=np.dot(phi[31*i+1:32+31*i],get_Ah(phi,p,h-i-1))

    elif h==0:
        Ah=np.identity(31) 
    elif h<0:
        Ah=np.zeros((31,31)) 
    return Ah

def get_result(PHI,dispflag):
    PHI_std=PHI/sum(PHI) #PHI为波动指数矩阵
    N=sum(sum(PHI_std))
    sum_SgH=0
    Si_gH=[] #方向性溢出效应
    S_igH=[] #方向性溢入效应
    Si_gH_add_self=[] #总溢出指数
    SijgH=np.zeros((31,31)) #两两净波动溢出
    for i in range(31):
        sum_S_igH=0
        sum_Si_gH=0
        for j in range(31):
            if abs(i-j): #i!=j
                sum_SgH+=PHI_std[i][j]
                sum_Si_gH+=PHI_std[i][j]
                sum_S_igH+=PHI_std[j][i]
                SijgH[i][j]=100*(sum_Si_gH-sum_S_igH)/N
        Si_gH.append(100*sum_Si_gH/N)
        Si_gH_add_self.append(Si_gH[i]+PHI[i][i])
        S_igH.append(100*sum_S_igH/N)
    SgH=100*sum_SgH/N
    SigH=list(np.array(Si_gH)-np.array(S_igH))
    return SgH,Si_gH,S_igH,SigH,SijgH,Si_gH_add_self

def my_VAR(Data,Start,End,p=10,H=4,dispflag=1):   
    VARModel, result=VARFitter(Data,lags=10)
    phi=result.params #对应的各个变量各滞后阶数的列表表示
    '''
    参数设置
    '''
    # p=10    #滞后阶数Start,End
    # H=4     #前向预测步数
    y=Data.values[Start:End]
    #y=Data.values[End-10:End]

    #第j个方程的标准差 sigma_jj
    sigma_jj=result.stderr_dt
    sigma_jj=sigma_jj.reshape(31,)

    # 预测值与实际值误差方差的协方差矩阵 THETA
    forcs=result.forecast(y=y,steps=H)
    real_value=Data.values[End:End+H]
    force_err=(forcs-real_value).T**2
    THETA=np.cov(force_err) #协方差矩阵

    # KPPS方差分解
    PHI=np.zeros((31,31)) #设一个风险溢出矩阵
    for i in range(31):
        for j in range(31):
            sum1=0
            sum2=0
            for h in range(H):
                tempAh=get_Ah(phi,p,h+1) #tempAh为系数矩阵,h为range（0-H）的值
                sum1+=(np.dot(tempAh[i],THETA[:][j]))**2
                sum2+=np.dot(np.dot(tempAh[i],THETA),tempAh.T[:][i])
            PHI[i][j]=sigma_jj[j]*sum1/sum2 #求出风险溢出矩阵PHI

    # 计算波动指数
    # SgH,Si_gH,S_igH,SigH,SijgH=get_result(PHI)
    return get_result(PHI,dispflag)
    #return PHI

#导入数据
Data=pd.read_csv('11.csv',header=0,index_col=0)
Data.head()

#滚动窗口
Windows=60
Length=200

SgH_all=[]
Si_gH_all=[]
S_igH_all=[]
SigH_all=[]
SijgH_all=[]
xTicks=[]
Si_gH_add_self_all=[]
rb_out=[]
ri_out=[]
rs_out=[]
rb_in=[]
ri_in=[]
rs_in=[]  
RB1=0
RB2=0
RI1=0
RI2=0
RS1=0
RS2=0
for i in range(57):
    Start=Windows*i
    End=Windows*i+Length
    SgH,Si_gH,S_igH,SigH,SijgH,Si_gH_add_self=my_VAR(Data,Start,End)
    for i in range(13):
        RB1+=Si_gH[i]
    rb_out.append(RB1) #13家银行风险溢出
    for i in range(13,16):
        RI1+=Si_gH[i]
    ri_out.append(RI1) #3家保险风险溢出
    for i in range(16,31):
        RS1+=Si_gH[i]
    rs_out.append(RS1) #15家证券风险溢出
    
    for i in range(13):
        RB2+=S_igH[i]
    rb_in.append(RB2) #13家银行风险溢入
    for i in range(13,16):
        RI2+=S_igH[i]
    ri_in.append(RI2) #3家保险风险溢入
    for i in range(16,31):
        RS2+=S_igH[i]
    rs_in.append(RS2) #15家证券风险溢入

    SgH_all.append(SgH) #总波动溢出指数
    Si_gH_all.append(Si_gH) #方向性溢出效应
    S_igH_all.append(S_igH) #方向性溢入效应
    SigH_all.append(SigH) #净波动溢出
    SijgH_all.append(SijgH)
    Si_gH_add_self_all.append(Si_gH_add_self) #各个变量的总溢出指数
Si_gH_all=np.array(Si_gH_all)
S_igH_all=np.array(S_igH_all)

#算出每个公司每个时间段的净溢出
net=[] 
for i in range(31):
    net1=[]
    for j in range(57):
        Net=Si_gH_all[:,i][j]-S_igH_all[:,i][j]
        net1.append(Net)
    net.append(net1)
net=np.array(net)   
net=net.T


#各个计算好的数据导出至excel，将dataframe保存为excel的代码
#31个公司各自的溢出效应
Si_df=pd.DataFrame(Si_gH_all)
Si_df.to_excel('out.xls',index=time,header=namelist)
#31个公司各自的溢入效应
S_igH_df=pd.DataFrame(S_igH_all)
S_igH_df.to_excel('in.xls',index=time,header=namelist)
#31个公司各自的净溢出效应
net_df=pd.DataFrame(net)
net_df=net_df.T
net_df.to_excel('net.xls',index=time,header=namelist)





#画出31个公司的31张图,且设置的双轴
namelist=['平安银行','交通银行','工商银行','中国银行','中信银行','宁波银行','浦发银行','华夏银行','民生银行',
         '招商银行','南京银行','兴业银行','北京银行','中国太保','中国平安','中国人寿','东北证券','华创阳安','西南证券',
          '华鑫股份','海通证券','哈投股份','太平洋','锦龙股份','国元证券','国海证券','广发证券','长江证券','中信证券',
          '湘财股份','国金证券'] 
#设置x轴的时间标签  
a=list(Data.index)
time=[]
for i in range(200,len(a)):
    if (i-200)%60==0:
        time.append(a[i])
        
Si_gH_all=np.array(Si_gH_all)
S_igH_all=np.array(S_igH_all)
       

for i in range(31):
    plt.rcParams['font.sans-serif']=['SimHei']
    matplotlib.rcParams['axes.unicode_minus']=False #解决坐标轴负号不能正常显示的问题
    fig = plt.figure(figsize=(16,8),dpi=80)
    ax1 = fig.add_subplot(111)
    plt.grid(visible=True, axis='both')
    plt.xticks(range(0,len(time),2),rotation=90)
    ax1.plot(time,SgH_all,label='growth',linestyle='-',color='k')#总波动溢出效应 
    ax1.plot(S_igH_all[:,i],label='in',linestyle=':',marker='+',color='k')#方向性波动溢入效应
    ax1.set_xlabel('时间')
    ax1.set_ylabel('波动')
    ax1.legend(loc=0)
    ax2 = ax1.twinx()
    ax2.plot(Si_gH_all[:,i],label='out',linestyle=':',marker='*',color='k')#方向性波动溢出效应
    ax2.plot(net[:,i],label='net',linestyle='-.',color='k')#净波动溢出效应
    plt.title(namelist[i]+'的金融风险动态研究')
    ax2.legend(loc=1)
    #plt.xticks(x,time,rotation=80)
   #从第一个数开始，一共显示原始横坐标长度len(time)个数，显示间隔为4，即每隔四个数显示一次横坐标
    plt.savefig(namelist[i]+'.jpg',bbox_inches='tight')
    #在保存图形是加入bbox_inches参数,解决坐标轴标签太长导致显示不全的问题
    plt.show()

