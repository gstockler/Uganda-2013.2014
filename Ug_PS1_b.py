#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 12:39:00 2019

@author: gabi
"""

import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

pd.options.display.float_format = '{:,.2f}'.format

from data_functions_albert import gini

from data_functions_albert import plot_cond_log_distr

#os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/UGA_2013_14_data')
os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/')

# =============================================================================
#               UGANDA 2013-2014 SURVEY DATA ANALYSIS
#
#                 DEVELOPMENT PS1 - LABOR SUPPLY
# =============================================================================

#%%
###############################################################################


"""

                Q2 & Q3 - Labor Supply Analysis & Inequality
                        
    In this file I (re)do the Problem Set focusing on labor supply variable
    Extensive margin is work or not per HouseHold
    Intesive margin is hours worked per year per HouseHold

"""

###                             LABOR DATA

# where hh_work is the extensive margin and hh_hour the intensive margin

labor = pd.read_csv("labor_supply.csv")
#data_labor = pd.read_csv("cwi_data.csv")
data_labor = pd.read_csv("data_13_14.csv")
data_labor = pd.merge(data_labor, labor, on="hh", how="left")
data_labor = data_labor.drop_duplicates(subset=['hh'], keep=False)
data_labor = data_labor[["hh","ctotal","inctotal","wtotal","age","sex","classeduc","region","urban","hh_work","hh_hour","hh_hour_prop"]]
del labor

#%% 2.1) Average Labor Supply - rural/urban

###                         Full sample

work_total = len(data_labor[data_labor.hh_work == 1]) # how many HH are working
hours_total = np.array(data_labor[["hh_hour"]].sum())[0] # total hours worked in the sample

print(work_total,hours_total)

###                     Urban vs Rural

## Urban general statistics
urban_sum = data_labor[(data_labor['urban'] == 1)]

# extensive margin
ext_urban = len(urban_sum[urban_sum.hh_work == 1])
ext_urban_prop = (ext_urban/work_total)*100

# intensive margin
int_urban = np.array(urban_sum[["hh_hour"]].sum())[0]
int_urban_mean = np.array(urban_sum[["hh_hour"]].mean())[0]
int_urban_prop = (int_urban/hours_total)*100

print(ext_urban,ext_urban_prop,int_urban,int_urban_mean,int_urban_prop )

## Rural
rural_sum = data_labor[(data_labor['urban'] == 0)]

# extensive margin
ext_rural = len(rural_sum[rural_sum.hh_work == 1])
ext_rural_prop = (ext_rural/work_total)*100

# intensive margin
int_rural = np.array(rural_sum[["hh_hour"]].sum())[0]
int_rural_mean = np.array(rural_sum[["hh_hour"]].mean())[0]
int_rural_prop = (int_rural/hours_total)*100

print(ext_rural,ext_rural_prop,int_rural,int_rural_mean,int_rural_prop )

del  ext_urban,ext_urban_prop,int_urban,int_urban_mean,int_urban_prop,ext_rural,ext_rural_prop,int_rural,int_rural_mean,int_rural_prop

#%% 2.2)  Histogram, variance (logs) - rural/urban

###                         Full sample

sns.set(color_codes=True)

## Histogram

# extensive
hist_ext=data_labor.hist(column='hh_work')
plt.title('Labor Supply - Extensive Margin', ha='center', fontsize='large')
plt.xlabel('work or not')
plt.ylabel('number of individuals')

hist_intt=data_labor.hist(column='hh_hour')
plt.title('Labor Supply - Intensive Margin', ha='center', fontsize='large')
plt.xlabel('hours per week')
plt.ylabel('number of individuals')

# Variance
data_labor.loc[data_labor.hh_hour > 0, "logg"] = data_labor['hh_hour']
data_labor['logg']=np.log(data_labor[['logg']])
full_var = data_labor[['hh_work','logg']].var()

print(full_var)
del full_var

###                          Rural vs Urban

# Histograms
sns.set(color_codes=True)

hist_ext=data_labor.hist(column='hh_work',by='urban')
plt.suptitle('Labor Supply - Extensive Margin: Rural vs. Urban', x=0.5, y=1.05, ha='center', fontsize='large')
plt.xlabel('work or not')
plt.ylabel('number of individuals')

hist_int=data_labor.hist(column='hh_hour',by='urban')
plt.suptitle('Labor Supply - Intensive Margin: Rural vs. Urban', x=0.5, y=1.05, ha='center', fontsize='large')
plt.xlabel('hours per week')
plt.ylabel('number of individuals')
plt.show()

del hist_ext, hist_w, hist_int

# Variances

urban_var = data_labor[(data_labor['urban'] == 1)].var()
print(urban_var[["hh_work","logg"]])

rural_var = data_labor[(data_labor['urban'] == 0)].var()
print(urban_var[["hh_work","logg"]])

del rural_var, urban_var

# Distribution

dist=data_labor[['hh_work','hh_hour','urban','ctotal','wtotal','inctotal']]
dist=dist.fillna(0)

target_0 = dist.loc[dist['urban'] == 0]
target_1 = dist.loc[dist['urban'] == 1]

sns.distplot(target_0[['hh_work']],hist=False,label='Rural')
sns.distplot(target_1[['hh_work']], hist=False,label='Urban')
plt.title('Extensive Labor Supply Inequality')
plt.ylabel('Density')
sns.plt.show()

sns.distplot(target_0[['hh_hour']], hist=False, label='Rural')
sns.distplot(target_1[['hh_hour']], hist=False,label='Urban')
plt.title('Intensive Labor Supply Inequality')
plt.ylabel('Density')
sns.plt.show()

del dist

#%% 2.3) Labor Supply Joint cross-sectional behavior

###                  Sumary Statistics

# Full sample
stats = data_labor.describe()
print(stats[["hh_work","hh_hour"]])

# rural
target_0.describe()

# urban
target_1.describe()

###                      Correlation

# ful sample
data_labor[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

# rural
target_0[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

# urban
target_1[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

del target_1, target_0


#%% 1.4) CIW Livecycle: level, inequality, covariances

###          Lifecycle variables: only considering 20-65

data_labor[['age']].describe()

cage = np.zeros((46,2))
ca= []
j= list(range(0,46))
for i in range(20,66):
    ca = data_labor.loc[data_labor['age'] == i]
    j= 20 -i
    cage[j,0]=ca['hh_work'].mean()
    cage[j,1]=ca['hh_hour'].mean()
    del ca
        
ages = list(range(20,66))

###                          CIW levels

plt.plot(ages,cage[:,0])
plt.xlabel('age')
plt.ylabel('work or not')
plt.title('Labor Supply Extensive Margin Life-Cycle (20-65 years)')

plt.plot(ages,cage[:,1])
plt.xlabel('age')
plt.ylabel('hours per week')
plt.title('Labor Supply Intesive Margin Life-Cycle (20-65 years)')

###                         Inequality - Gini

# the greater (close to 1), more inequality

# Full sample
g = data_labor[["hh_work","hh_hour","age"]]
g=g.fillna(0)
 
w_array = np.array(g['hh_work'])
gini_work = gini(w_array)

h_array = np.array(g['hh_hour'])
gini_hour = gini(h_array)

print(gini_work, gini_hour)

del gini_work, gini_hour

# By Ages

gw= np.zeros((46,1))
gh=np.zeros((46,1))

j= list(range(0,46))
for i in range(20,66):
    ca = g.loc[g['age'] == i]
    j= 20 -i
    w_array = np.array(ca['hh_work'])
    gw[j,0]=gini(w_array)
    h_array = np.array(ca['hh_hour'])
    gh[j,0]=gini(h_array)
    del ca, w_array, h_array
    
plt.plot(ages,gw)
plt.xlabel('age')
plt.ylabel('extensive labor supply gini')
plt.title('Ext LS-Inequality over the Life-Cycle')

plt.plot(ages,gh)
plt.xlabel('age')
plt.ylabel('extensive labor supplygini')
plt.title('Int LS-Inequality over the Life-Cycle')

plt.plot(ages,gw)
plt.plot(ages,gh)
plt.legend('EI')
plt.xlabel('age')
plt.ylabel('gini')
plt.title('Labor Supply Inequality over the Life-Cycle')
plt.show()

###    Covariances 

cov=[]                                                   
for i in range(20,66):
    tryt=g.loc[g['age'] == i].cov()
    tryt=np.array(tryt)
    cov.append(tryt[0:2,0:2])
    del tryt

cov_wh = np.zeros((46,1))
for j in range(0,46):
    cov_wh[j,0] = cov[j][0,1] 

plt.plot(ages,cov_wh)
plt.title('Lifecycile Labor Suply Covariance')
del cov_wh

#%% 1.5) Income Ranking and CW distributions conditional on I

###                         Percentiles based on Income 


pct = data_labor['inctotal'].quantile([0.01,0.05, 0.1, 0.2, 0.4,0.6,0.8,0.9,0.95,0.99,1])
pct=np.array(pct)
 
# sort data (lowest to highest income)   
datasort=data_labor.sort_values('inctotal')
l_rank=np.zeros((11,2))

# get variables per percentile    
tt = datasort.loc[datasort['inctotal'] <= pct[0]]
l_rank[0,0]=tt['hh_work'].mean()
l_rank[0,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[0] < datasort['inctotal']) & (datasort['inctotal'] <= pct[1])]
l_rank[1,0]=tt['hh_work'].mean()
l_rank[1,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[1] < datasort['inctotal']) & (datasort['inctotal'] <= pct[2])]
l_rank[2,0]=tt['hh_work'].mean()
l_rank[2,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[datasort['inctotal'] <= pct[3]]
l_rank[3,0]=tt['hh_work'].mean()
l_rank[3,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[3] < datasort['inctotal']) & (datasort['inctotal'] <= pct[4])]
l_rank[4,0]=tt['hh_work'].mean()
l_rank[4,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[4] < datasort['inctotal']) & (datasort['inctotal'] <= pct[5])]
l_rank[5,0]=tt['hh_work'].mean()
l_rank[5,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[5] < datasort['inctotal']) & (datasort['inctotal'] <= pct[6])]
l_rank[6,0]=tt['hh_work'].mean()
l_rank[6,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[datasort['inctotal'] > pct[6]]
l_rank[7,0]=tt['hh_work'].mean()
l_rank[7,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[7] < datasort['inctotal']) & (datasort['inctotal'] <= pct[8])]
l_rank[8,0]=tt['hh_work'].mean()
l_rank[8,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[(pct[8] < datasort['inctotal']) & (datasort['inctotal'] <= pct[9])]
l_rank[9,0]=tt['hh_work'].mean()
l_rank[9,1]=tt['hh_hour'].mean()
del tt

tt = datasort.loc[datasort['inctotal'] > pct[9]]
l_rank[10,0]=tt['hh_work'].mean()
l_rank[10,1]=tt['hh_hour'].mean()
del tt

# Proportions 
l_rank_share=np.zeros((11,2))
l_rank_share[:,0]=(l_rank[:,0]/work_total)*100
l_rank_share[:,1]=(l_rank[:,1]/hours_total)*100


#%%
"""

                Q3 - Labor Supply Inequality across Space
                                BY ZONE

"""

###                         Regions: 1-4

# separate the data

data=data_labor
data[['region']].describe()

r1 = data.loc[data['region'] == 1]
r1=r1[['inctotal','hh_work','hh_hour']]
r1.describe()

r2 = data.loc[data['region'] == 2]
r2=r2[['inctotal','hh_work','hh_hour']]
r2.describe()

r3 = data.loc[data['region'] == 3]
r3=r3[['inctotal','hh_work','hh_hour']]
r3.describe()

r4 = data.loc[data['region'] == 4]
r4=r4[['inctotal','hh_work','hh_hour']]
r4.describe()

reg = np.zeros((4,3))
rr = []

for i in range(1,5):
    rr = data.loc[data['region'] == i]
    reg[i-1,0]=rr['hh_work'].mean()
    reg[i-1,1]=rr['hh_hour'].mean()
    reg[i-1,2]=rr['inctotal'].mean()
    del rr

#%% 3.1) Level

## Region 1
plt.scatter(r1['inctotal'],r1['hh_hour'])
plt.xlabel('income level')
plt.ylabel('intensive ')
plt.title('LS vs Income - Region 1' )
plt.show()
plt.scatter(r1['inctotal'],r1['hh_work'])
plt.xlabel('income level')
plt.ylabel('extensive ')
plt.title('LS vs Income - Region 1' )
plt.show()

## Region 2
plt.scatter(r2['inctotal'],r2['hh_hour'])
plt.xlabel('income level')
plt.ylabel('intensive ')
plt.title('LS vs Income - Region 2' )
plt.show()
plt.scatter(r2['inctotal'],r2['hh_work'])
plt.xlabel('income level')
plt.ylabel('extensive ')
plt.title('LS vs Income - Region 2' )
plt.show()

## Region 3
plt.scatter(r3['inctotal'],r3['hh_hour'])
plt.xlabel('income level')
plt.ylabel('intensive ')
plt.title('LS vs Income - Region 3' )
plt.show()
plt.scatter(r3['inctotal'],r3['hh_work'])
plt.xlabel('income level')
plt.ylabel('extensive ')
plt.title('LS vs Income - Region 3' )
plt.show()

## Region 4
plt.scatter(r4['inctotal'],r4['hh_hour'])
plt.xlabel('income level')
plt.ylabel('intensive ')
plt.title('LS vs Income - Region 4' )
plt.show()
plt.scatter(r4['inctotal'],r4['hh_work'])
plt.xlabel('income level')
plt.ylabel('extensive ')
plt.title('LS vs Income - Region 4' )
plt.show()

## All regions comparison
plt.scatter(r1[['inctotal']],r1[['hh_work']])
plt.scatter(r2[['inctotal']],r2[['hh_work']])
plt.scatter(r3[['inctotal']],r3[['hh_work']])
plt.scatter(r4[['inctotal']],r4[['hh_work']])
plt.title('Extensive LS vs Income across Regions')
plt.legend('1234')
plt.xlabel('income level')
plt.ylabel('labor supply ')
plt.show()

plt.scatter(r1[['inctotal']],r1[['hh_hour']])
plt.scatter(r2[['inctotal']],r2[['hh_hour']])
plt.scatter(r3[['inctotal']],r3[['hh_hour']])
plt.scatter(r4[['inctotal']],r4[['hh_hour']])
plt.title('Intensive LS vs Income across Regions')
plt.legend('1234')
plt.xlabel('income level')
plt.ylabel('labor supply ')
plt.show()


#%% 3.2) Inequality 

r1=r1.fillna(0)
r2=r2.fillna(0)
r3=r3.fillna(0)
r4=r4.fillna(0)

ri = [r1['inctotal'].mean(),r2['inctotal'].mean(),r3['inctotal'].mean() ,r4['inctotal'].mean()]

## Gini for consumption

c1 = np.array(r1['hh_work'])
g1c = gini(c1)

c2 = np.array(r2['hh_work'])
g2c = gini(c2)

c3 = np.array(r3['hh_work'])
g3c = gini(c3)

c4 = np.array(r4['hh_work'])
g4c = gini(c4)

del c1,c2,c3,c4

plt.scatter(ri,[g1c,g2c,g3c,g4c])
plt.title('Mean Income vs Gini Extensive LS - by Region')

## Gini for wealth

c1 = np.array(r1['hh_hour'])
g1w = gini(c1)

c2 = np.array(r2['hh_hour'])
g2w = gini(c2)

c3 = np.array(r3['hh_hour'])
g3w = gini(c3)

c4 = np.array(r4['hh_hour'])
g4w = gini(c4)

del c1,c2,c3,c4

plt.scatter(ri,[g1w,g2w,g3w,g4w])
plt.title('Mean Income vs Gini Intensive LS - by Region')


#%% 3.3) Covariances

# Region 1
r1.cov()
r1.corr()

# Region 2
r2.cov()
r2.corr()

# Region 3
r3.cov()
r3.corr()

# Region 4
r4.cov()
r4.corr()

#%%
###############################################################################

"""

                Q2.2- Labor Supply Inequality by Sex

"""

# I suggest clearing the working space first

# Then, call the data set again

labor = pd.read_csv("labor_supply.csv")
#data_labor = pd.read_csv("cwi_data.csv")
data_labor = pd.read_csv("data_13_14.csv")
data_labor = pd.merge(data_labor, labor, on="hh", how="left")
data_labor = data_labor.drop_duplicates(subset=['hh'], keep=False)
data_labor = data_labor[["hh","ctotal","inctotal","wtotal","age","sex","classeduc","region","urban","hh_work","hh_hour","hh_hour_prop"]]
del labor


#%% 2.1) Average Labor Supply - men(1)/women

###                         Full sample

work_total = len(data_labor[data_labor.hh_work == 1]) # how many HH are working
hours_total = np.array(data_labor[["hh_hour"]].sum())[0] # total hours worked in the sample

print(work_total,hours_total)

###                     Men vs Woman

## Men general statistics
men_sum = data_labor[(data_labor['sex'] == 1)]

# extensive margin
ext_men = len(men_sum[men_sum.hh_work == 1])
ext_men_prop = (ext_men/work_total)*100

# intensive margin
int_men = np.array(men_sum[["hh_hour"]].sum())[0]
int_men_mean = np.array(men_sum[["hh_hour"]].mean())[0]
int_men_prop = (int_men/hours_total)*100

print(ext_men,ext_men_prop,int_men,int_men_mean,int_men_prop )

## Women
wom_sum = data_labor[(data_labor['sex'] == 2)]

# extensive margin
ext_wom = len(wom_sum[wom_sum.hh_work == 1])
ext_wom_prop = (ext_wom/work_total)*100

# intensive margin
int_wom = np.array(wom_sum[["hh_hour"]].sum())[0]
int_wom_mean = np.array(wom_sum[["hh_hour"]].mean())[0]
int_wom_prop = (int_wom/hours_total)*100

print(ext_wom,ext_wom_prop,int_wom,int_wom_mean,int_wom_prop )

del ext_men,ext_men_prop,int_men,int_men_mean,int_men_prop ,ext_wom,ext_wom_prop,int_wom,int_wom_mean,int_wom_prop
#%% 2.2)  Histogram

###                         Full sample


###                          Men vs Woman

# Histograms
sns.set(color_codes=True)

hist_ext=data_labor.hist(column='hh_work',by='sex')
plt.suptitle('Labor Supply - Extensive Margin: Men vs. Woman', x=0.5, y=1.05, ha='center', fontsize='large')
plt.xlabel('work or not')
plt.ylabel('number of individuals')

hist_int=data_labor.hist(column='hh_hour',by='sex')
plt.suptitle('Labor Supply - Intensive Margin: Men vs. Woman', x=0.5, y=1.05, ha='center', fontsize='large')
plt.xlabel('hours per week')
plt.ylabel('number of individuals')
plt.show()

del hist_ext, hist_int

# Distribution

dist=data_labor[['hh_work','hh_hour','sex','ctotal','wtotal','inctotal']]
dist=dist.fillna(0)

target_0 = dist.loc[dist['sex'] == 1]
target_1 = dist.loc[dist['sex'] == 2]

sns.distplot(target_0[['hh_work']],hist=False,label='Men')
sns.distplot(target_1[['hh_work']], hist=False,label='Woman')
plt.title('Extensive Labor Supply Inequality')
plt.ylabel('Density')
sns.plt.show()

sns.distplot(target_0[['hh_hour']], hist=False, label='Men')
sns.distplot(target_1[['hh_hour']], hist=False,label='Woman')
plt.title('Intensive Labor Supply Inequality')
plt.ylabel('Density')
sns.plt.show()

del dist

#%% 2.3) Labor Supply Joint cross-sectional behavior

###                  Sumary Statistics

# rural
target_0.describe()

# urban
target_1.describe()

###                      Correlation

# rural
target_0[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

# urban
target_1[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

del target_1, target_0

#%%
###############################################################################
"""

                Q2.3- Labor Supply Inequality by Education
                

"""

# I suggest clearing the working space first

# Then, call the data set again

labor = pd.read_csv("labor_supply.csv")
#data_labor = pd.read_csv("cwi_data.csv")
data_labor = pd.read_csv("data_13_14.csv")
data_labor = pd.merge(data_labor, labor, on="hh", how="left")
data_labor = data_labor.drop_duplicates(subset=['hh'], keep=False)
data_labor = data_labor[["hh","ctotal","inctotal","wtotal","age","sex","classeduc","region","urban","hh_work","hh_hour","hh_hour_prop"]]
del labor


#%% 2.1) Average Labor Supply - educ

###                         Full sample

work_total = len(data_labor[data_labor.hh_work == 1]) # how many HH are working
hours_total = np.array(data_labor[["hh_hour"]].sum())[0] # total hours worked in the sample

print(work_total,hours_total)

###                         Educ Groups

## Less than primary school (NP)

np_sum = data_labor[(data_labor['classeduc'] == 10)]
np_sum=np_sum.append(data_labor[(data_labor['classeduc'] == 99)] )

len(np_sum)
# extensive margin
ext_np = len(np_sum[np_sum.hh_work == 1])
ext_np_prop = (ext_np/work_total)*100

# intensive margin
int_np = np.array(np_sum[["hh_hour"]].sum())[0]
int_np_mean = np.array(np_sum[["hh_hour"]].mean())[0]
int_np_prop = (int_np/hours_total)*100

## Primary School completed
p_sum = data_labor[(data_labor['classeduc'] == 11)]
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 12)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 13)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 14)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 15)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 16)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 17)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 21)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 22)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 23)] )
p_sum=p_sum.append(data_labor[(data_labor['classeduc'] == 41)] )

len(p_sum)
# extensive margin
ext_p = len(p_sum[p_sum.hh_work == 1])
ext_p_prop = (ext_p/work_total)*100

# intensive margin
int_p = np.array(p_sum[["hh_hour"]].sum())[0]
int_p_mean = np.array(p_sum[["hh_hour"]].mean())[0]
int_p_prop = (int_p/hours_total)*100

## Secondary School or higher
s_sum = data_labor[(data_labor['classeduc'] == 31)]
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 32)] )
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 33)] )
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 34)] )
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 35)] )
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 36)] )
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 51)] )
s_sum=s_sum.append(data_labor[(data_labor['classeduc'] == 61)] )

len(s_sum)
# extensive margin
ext_s= len(s_sum[s_sum.hh_work == 1])
ext_s_prop = (ext_s/work_total)*100

# intensive margin
int_s = np.array(s_sum[["hh_hour"]].sum())[0]
int_s_mean = np.array(s_sum[["hh_hour"]].mean())[0]
int_s_prop = (int_s/hours_total)*100
#%% 2.2)  Histogram

###                         Full sample


###                          Educ Groups

data_labor.loc[data_labor.classeduc == 10, "educgroup"] = 0
data_labor.loc[data_labor.classeduc == 99, "educgroup"] = 0
data_labor.loc[data_labor.classeduc == 11, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 12, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 13, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 14, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 15, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 16, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 17, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 21, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 22, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 23, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 41, "educgroup"] = 1
data_labor.loc[data_labor.classeduc == 31, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 32, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 33, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 34, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 35, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 35, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 51, "educgroup"] = 2
data_labor.loc[data_labor.classeduc == 61, "educgroup"] = 2

# Histograms
sns.set(color_codes=True)

hist_ext=data_labor.hist(column='hh_work',by='educgroup')
plt.suptitle('Labor Supply - Extensive Margin: Education Level', x=0.5, y=1.05, ha='center', fontsize='large')
plt.xlabel('work or not')
plt.ylabel('number of individuals')

hist_int=data_labor.hist(column='hh_hour',by='educgroup')
plt.suptitle('Labor Supply - Intensive Margin: Education Level', x=0.5, y=1.05, ha='center', fontsize='large')
plt.xlabel('hours per week')
plt.ylabel('number of individuals')
plt.show()

del hist_ext, hist_int

# Distribution

dist=data_labor[['hh_work','hh_hour','educgroup','ctotal','wtotal','inctotal']]
dist=dist.fillna(0)

target_0 = dist.loc[dist['educgroup'] == 0]
target_1 = dist.loc[dist['educgroup'] == 1]
target_2 = dist.loc[dist['educgroup'] == 2]

sns.distplot(target_0[['hh_work']],hist=False,label='Less than Primary School')
sns.distplot(target_1[['hh_work']], hist=False,label='Completed Primary School')
sns.distplot(target_2[['hh_work']], hist=False,label='Secondary School or higher')
plt.title('Extensive Labor Supply Inequality')
plt.ylabel('Density')
sns.plt.show()

sns.distplot(target_0[['hh_hour']], hist=False, label='Less than Primary School')
sns.distplot(target_1[['hh_hour']], hist=False,label='Completed Primary School')
sns.distplot(target_2[['hh_hour']], hist=False,label='Secondary School or higher')
plt.title('Intensive Labor Supply Inequality')
plt.ylabel('Density')
sns.plt.show()

del dist

#%% 2.3) Labor Supply Joint cross-sectional behavior

###                  Sumary Statistics

# NP
target_0.describe()

# P
target_1.describe()

# S
target_2.describe()

###                      Correlation

# NP
target_0[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

# P
target_1[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

# S
target_2[['hh_hour','hh_work','ctotal','inctotal','wtotal']].corr()

del target_1, target_0, target_2