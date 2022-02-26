# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:48:40 2020

@author: YANGQINGYUAN
"""
import pandas as pd 
import numpy as np
import os
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
from scipy import stats
from jqdatasdk import *
auth('17376507314','Xiaoshuai0928')

def Cal_ambiguity(futuredf):
    '''
    Description
    -----------
    传入期货分钟级数据，返回模糊性序列
    
    Parameters
    ----------
    futuredf : n*1 dataframe
        index: 5min时间序列
        columns: 收益率

    Returns
    -------
    None.

    '''
    # 读取数据并调整至5min
    futuredf = pd.DataFrame(futuredf.close_ret)
    futuredf = futuredf - 1
    futuredf.index = pd.to_datetime(futuredf.index)
    futuredf = futuredf.resample('5min').sum()
    
    # 得到当月数据，对月循环
    tempmonth = futuredf.resample('m').count().index
    mhodf = pd.DataFrame()
    for tempmonthi in tempmonth:
        monthdf = futuredf[tempmonthi.strftime('%Y-%m')]
        monthdf = monthdf[monthdf.index.hour.isin([9,10,11,13,14])]
        monthdf[(monthdf.index.hour==10)&(monthdf.index.minute>10)&(monthdf.index.minute<30)] = np.nan
        monthdf[(monthdf.index.hour==11)&(monthdf.index.minute>25)] = np.nan
        monthdf[(monthdf.index.hour==13)&(monthdf.index.minute<30)] = np.nan
        monthdf = monthdf.dropna()
        ## 得到当天数据，对天循环
        tempdate = monthdf.resample('d').count().index
        promonthdf = pd.DataFrame()
        rlist = []
        temp = -0.06
        while temp < 0.061:
            rlist.append(temp)
            temp += 0.00001
        for tempdatei in tempdate:
            daydf = monthdf.loc[tempdatei.strftime('%Y%m%d')]
            ## 对r循环，得到每天对应r的均值标准差
            prodaydf = pd.DataFrame(columns = ['r','mean','sigma'])
            tempmean = daydf.mean()[0]
            tempstd = daydf.std()[0]
            tempdf = pd.DataFrame()
            tempdf.insert(0,'r',rlist)
            tempdf.insert(1,'mean',tempmean)
            tempdf.insert(2,'sigma',tempstd)
            prodaydf = prodaydf.append(tempdf)
            prodaydf.insert(0,'date',tempdatei)
            prodaydf = prodaydf.replace(0,np.nan)
            prodaydf.dropna(inplace=True)
            cumprob = stats.norm.cdf(prodaydf.iloc[:,1],prodaydf.iloc[:,2],prodaydf.iloc[:,3])
            prodaydf.insert(4,'cumprob',cumprob)
            prodaydf.insert(5,'prob',prodaydf.cumprob.diff())
            try:
                prodaydf.iloc[0,5] = cumprob[0]
            except:
                pass
            promonthdf = promonthdf.append(prodaydf)
        meanprob = pd.DataFrame(promonthdf.groupby('r').mean().iloc[:,-1])
        varprob = pd.DataFrame(promonthdf.groupby('r').std().iloc[:,-1] ** 2)
        mho2 = 1 / 0.00001 / (1-0.00001) * np.dot(meanprob.values.T,varprob.values)[0,0]
        tempdf = pd.DataFrame([mho2],index=[tempmonthi],columns=['ambiguity'])
        mhodf = mhodf.append(tempdf)
    return mhodf