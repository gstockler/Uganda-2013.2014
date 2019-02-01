#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 19:03:16 2019

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
#                           DEVELOPMENT PS1
# =============================================================================


"""

                Q1 & Q3 - CIW Analysis & Inequality
    
        In this file I do the Problem Set focusing on CIW variables

"""

#%% 1.1) Average CIW - rural/urban

cwi_data=pd.read_csv('cwi_data.csv')
data=pd.read_csv('data_13_14.csv')

###                     Full Sample

c_hh_mean=cwi_data['ctotal'].mean(axis=0)
inc_hh_mean=cwi_data['inctotal'].mean(axis=0)
w_hh_mean=cwi_data['wtotal'].mean(axis=0)

print(c_hh_mean,inc_hh_mean,w_hh_mean)

###                 Urban vs Rural areas

# Urban general statistics
urban_sum = cwi_data[(cwi_data['urban'] == 1)].mean()
print(urban_sum[["ctotal","inctotal","wtotal"]])

# Rural
rural_sum = cwi_data[(cwi_data['urban'] == 0)].mean()
print(rural_sum[["ctotal","inctotal","wtotal"]])

del rural_sum, urban_sum, c_hh_mean,inc_hh_mean,w_hh_mean

#%% 1.2)  Histogram, variance (logs) - rural/urban

###                         Full sample

sns.set(color_codes=True)

# Histogram
hist_ct=cwi_data.hist(column='ctotal')
plt.title('Consumption Inequality', ha='center', fontsize='large')
plt.xlabel('consumption levels')
plt.ylabel('number of individuals')

#sns.distplot(cdist[['ctotal']], hist=True)
#plt.title('Consumption Distribution')
#plt.ylabel('Density')

# Variance
cwi_data[['log_c','log_inc','log_w']] = np.log(cwi_data[['ctotal','inctotal','wtotal']])
full_var = cwi_data[['log_c','log_inc','log_w']].var()

print(full_var)
del full_var

###                          Rural vs Urban

# Histograms

sns.set(color_codes=True)
hist_c = cwi_data.hist(column='ctotal',by='urban')
plt.suptitle('Consumption Inequality: Rural vs. Urban', x=0.5, y=1.05, ha='center', fontsize='large')

hist_i = cwi_data.hist(column='inctotal',by='urban')
plt.suptitle('Income Inequality: Rural vs. Urban', x=0.5, y=1.05, ha='center', fontsize='large')

hist_w = cwi_data.hist(column='wtotal',by='urban')
plt.suptitle('Wealth Inequality: Rural vs. Urban', x=0.5, y=1.05, ha='center', fontsize='large')

del hist_c, hist_w, hist_i

# Variances

urban_var = cwi_data[(cwi_data['urban'] == 1)].var()
print(urban_var[["log_c","log_inc","log_w"]])

rural_var = cwi_data[(cwi_data['urban'] == 0)].var()
print(rural_var[["log_c","log_inc","log_w"]])

del rural_var, urban_var

# Distribution

dist=cwi_data[['ctotal','inctotal','wtotal','urban']]
dist=dist.fillna(0)

target_0 = dist.loc[dist['urban'] == 0]
target_1 = dist.loc[dist['urban'] == 1]

sns.distplot(target_0[['ctotal']], hist=False, label='Rural')
sns.distplot(target_1[['ctotal']], hist=False,label='Urban')
plt.title('Consumption Inequality')
plt.ylabel('Density')
sns.plt.show()

sns.distplot(target_0[['inctotal']], hist=False, label='Rural')
sns.distplot(target_1[['inctotal']], hist=False,label='Urban')
plt.title('Income Inequality')
plt.ylabel('Density')
sns.plt.show()

sns.distplot(target_0[['wtotal']], hist=False, label='Rural')
sns.distplot(target_1[['wtotal']], hist=False,label='Urban')
plt.title('Wealth Inequality')
plt.ylabel('Density')
sns.plt.show()

del dist

#%% 1.3) CIW Joint cross-sectional behavior

###                  Sumary Statistics

# Full sample
stats = cwi_data.describe()
print(stats[["ctotal","inctotal","wtotal"]])

# rural
target_0.describe()

# urban
target_1.describe()

###                      Correlation

# ful sample
cwi_data[['ctotal','wtotal','inctotal']].corr()

# rural
target_0[['ctotal','wtotal','inctotal']].corr()

# urban
target_1[['ctotal','wtotal','inctotal']].corr()

del target_1, target_0

#%% 1.4) CIW Livecycle: level, inequality, covariances

###          Lifecycle variables: only considering 20-65

data[['age']].describe()

cage = np.zeros((46,3))
ca= []
j= list(range(0,46))
for i in range(20,66):
    ca = data.loc[data['age'] == i]
    j= 20 -i
    cage[j,0]=ca['ctotal'].mean()
    cage[j,1]=ca['inctotal'].mean()
    cage[j,2]=ca['wtotal'].mean()
    del ca
        
ages = list(range(20,66))

###                          CIW levels

plt.plot(ages,cage[:,0])
plt.xlabel('age')
plt.ylabel('level in US dollars')
plt.title('Consumption Life-Cycle (20-65 years)')

plt.plot(ages,cage[:,1])
plt.xlabel('age')
plt.ylabel('level in US dollars')
plt.title('Income Life-Cycle (20-65 years)')

plt.plot(ages,cage[:,2])
plt.xlabel('age')
plt.ylabel('level in US dollars')
plt.title('Wealth Life-Cycle (20-65 years)')

plt.plot(ages,np.log(cage[:,0]))
plt.plot(ages,np.log(cage[:,1]))
plt.plot(ages,np.log(cage[:,2]))
plt.legend('CIW ')
plt.xlabel('age')
plt.ylabel('log')
plt.title('CIW over the Life-Cycle')
plt.show()

###                         Inequality - Gini

# the greater (close to 1), more inequality

# Full sample
g = data[["ctotal","inctotal","wtotal","age"]]
g=g.fillna(0)
 
c_array = np.array(g['ctotal'])
gini_c = gini(c_array)

inc_array = np.array(g['inctotal'])
gini_inc = gini(inc_array)

w_array = np.array(g['wtotal'])
gini_w = gini(w_array)

del g

print(gini_c, gini_inc, gini_w)

# By Ages

gc= np.zeros((46,1))
gi=np.zeros((46,1))
gw=np.zeros((46,1))

j= list(range(0,46))
for i in range(20,66):
    ca = g.loc[g['age'] == i]
    j= 20 -i
    c_array = np.array(ca['ctotal'])
    gc[j,0]=gini(c_array)
    inc_array = np.array(ca['inctotal'])
    gi[j,0]=gini(inc_array)
    w_array = np.array(ca['wtotal'])
    gw[j,0]=gini(w_array)

    del ca, c_array, w_array, inc_array
    
plt.plot(ages,gc)
plt.xlabel('age')
plt.ylabel('consumption gini')
plt.title('C-Inequality over the Life-Cycle')

plt.plot(ages,gi)
plt.xlabel('age')
plt.ylabel('income gini')
plt.title('I-Inequality over the Life-Cycle')

plt.plot(ages,gw)
plt.xlabel('age')
plt.ylabel('wealth gini')
plt.title('W-Inequality over the Life-Cycle')

plt.plot(ages,gc)
plt.plot(ages,gi)
plt.plot(ages,gw)
plt.legend('CIW ')
plt.xlabel('age')
plt.ylabel('gini')
plt.title('CIW Inequality over the Life-Cycle')
plt.show()

###    Covariances 

cov=[]                                                   
for i in range(20,66):
    tryt=g.loc[g['age'] == i].cov()
    tryt=np.array(tryt)
    cov.append(tryt[0:3,0:3])
    del tryt

cov_ci = np.zeros((46,1))
for j in range(0,46):
    cov_ci[j,0] = cov[j][0,1] 

plt.plot(ages,cov_ci)
plt.tile('Lifecycile CI Covariance')
del cov_ci

cov_cw = np.zeros((46,1))
for j in range(0,46):
    cov_cw[j,0] = cov[j][0,2] 

plt.plot(ages,cov_cw)
plt.tile('Lifecycile CW Covariance')
del cov_cw

cov_wi = np.zeros((46,1))
for j in range(0,46):
    cov_wi[j,0] = cov[j][1,2] 
    
plt.plot(ages,cov_wi)
plt.tile('Lifecycile WI Covariance')
del cov_wi




#%% 1.5) Income Ranking and CW distributions conditional on I

###                         Percentiles based on Income 


pct = data_cwi['inctotal'].quantile([0.01,0.05, 0.1, 0.2, 0.4,0.6,0.8,0.9,0.95,0.99,1])
pct=np.array(pct)
 
# sort data (lowest to highest income)   
datasort=data_cwi.sort_values('inctotal')
cw_rank=np.zeros((11,2))

# get variables per percentile    
tt = datasort.loc[datasort['inctotal'] <= pct[0]]
cw_rank[0,0]=tt['ctotal'].sum()
cw_rank[0,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[0] < datasort['inctotal']) & (datasort['inctotal'] <= pct[1])]
cw_rank[1,0]=tt['ctotal'].sum()
cw_rank[1,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[1] < datasort['inctotal']) & (datasort['inctotal'] <= pct[2])]
cw_rank[2,0]=tt['ctotal'].sum()
cw_rank[2,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[datasort['inctotal'] <= pct[3]]
cw_rank[3,0]=tt['ctotal'].sum()
cw_rank[3,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[3] < datasort['inctotal']) & (datasort['inctotal'] <= pct[4])]
cw_rank[4,0]=tt['ctotal'].sum()
cw_rank[4,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[4] < datasort['inctotal']) & (datasort['inctotal'] <= pct[5])]
cw_rank[5,0]=tt['ctotal'].sum()
cw_rank[5,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[5] < datasort['inctotal']) & (datasort['inctotal'] <= pct[6])]
cw_rank[6,0]=tt['ctotal'].sum()
cw_rank[6,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[datasort['inctotal'] > pct[6]]
cw_rank[7,0]=tt['ctotal'].sum()
cw_rank[7,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[7] < datasort['inctotal']) & (datasort['inctotal'] <= pct[8])]
cw_rank[8,0]=tt['ctotal'].sum()
cw_rank[8,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[(pct[8] < datasort['inctotal']) & (datasort['inctotal'] <= pct[9])]
cw_rank[9,0]=tt['ctotal'].sum()
cw_rank[9,1]=tt['wtotal'].sum()
del tt

tt = datasort.loc[datasort['inctotal'] > pct[9]]
cw_rank[10,0]=tt['ctotal'].sum()
cw_rank[10,1]=tt['wtotal'].sum()
del tt

# Aggregate levels - since we will look at the proportions
c_agg=data_cwi['ctotal'].sum()
inc_agg=data_cwi['inctotal'].sum()
w_agg=data_cwi['wtotal'].sum()

# Proportions 
c_rank_share=np.zeros((11,2))
c_rank_share[:,0]=(cw_rank[:,0]/c_agg)*100
c_rank_share[:,1]=(cw_rank[:,1]/w_agg)*100


#%%
"""

                Q3 - Inequality across Space
                            BY ZONE

"""

###                         Regions: 1-4

# separate the data

data[['region']].describe()

r1 = data.loc[data['region'] == 1]
r1=r1[['inctotal','ctotal','wtotal']]
r1.describe()

r2 = data.loc[data['region'] == 2]
r2=r2[['inctotal','ctotal','wtotal']]
r2.describe()

r3 = data.loc[data['region'] == 3]
r3=r3[['inctotal','ctotal','wtotal']]
r3.describe()

r4 = data.loc[data['region'] == 4]
r4=r4[['inctotal','ctotal','wtotal']]
r4.describe()

reg = np.zeros((4,3))
rr = []

for i in range(1,5):
    rr = data.loc[data['region'] == i]
    reg[i-1,0]=rr['ctotal'].mean()
    reg[i-1,1]=rr['inctotal'].mean()
    reg[i-1,2]=rr['wtotal'].mean()
    del rr

#%% 3.1) Level

## Region 1
plt.scatter(r1['inctotal'],r1['ctotal'])
plt.xlabel('income level')
plt.ylabel('consumption level ')
plt.title('Consumption vs Income - Region 1' )
plt.show()
plt.scatter(r1['inctotal'],r1['wtotal'])
plt.xlabel('income level')
plt.ylabel('wealth level ')
plt.title('Wealth vs Income - Region 1' )
plt.show()

## Region 2
plt.scatter(r2['inctotal'],r2['ctotal'])
plt.xlabel('income level')
plt.ylabel('consumption level ')
plt.title('Consumption vs Income - Region 2' )
plt.show()
plt.scatter(r2['inctotal'],r2['wtotal'])
plt.xlabel('income level')
plt.ylabel('wealth level ')
plt.title('Wealth vs Income - Region 2' )
plt.show()

## Region 3
plt.scatter(r3['inctotal'],r3['ctotal'])
plt.xlabel('income level')
plt.ylabel('consumption level ')
plt.title('Consumption vs Income - Region 3' )
plt.show()
plt.scatter(r3['inctotal'],r3['wtotal'])
plt.xlabel('income level')
plt.ylabel('wealth level ')
plt.title('Wealth vs Income - Region 3' )
plt.show()

## Region 4
plt.scatter(r4['inctotal'],r4['ctotal'])
plt.xlabel('income level')
plt.ylabel('consumption level ')
plt.title('Consumption vs Income - Region 4' )
plt.show()
plt.scatter(r4['inctotal'],r4['wtotal'])
plt.xlabel('income level')
plt.ylabel('wealth level ')
plt.title('Wealth vs Income - Region 4' )
plt.show()

## All regions comparison
plt.scatter(r1[['inctotal']],r1[['ctotal']])
plt.scatter(r2[['inctotal']],r2[['ctotal']])
plt.scatter(r3[['inctotal']],r3[['ctotal']])
plt.scatter(r4[['inctotal']],r4[['ctotal']])
plt.title('Consumption vs Income across Regions')
plt.legend('1234')
plt.xlabel('income level')
plt.ylabel('consumption level ')
plt.show()

plt.scatter(r1[['inctotal']],r1[['wtotal']])
plt.scatter(r2[['inctotal']],r2[['wtotal']])
plt.scatter(r3[['inctotal']],r3[['wtotal']])
plt.scatter(r4[['inctotal']],r4[['wtotal']])
plt.title('Wealth vs Income across Regions')
plt.legend('1234')
plt.xlabel('income level')
plt.ylabel('wealth level ')
plt.show()

## All regions mean comparison
rc = [r1['ctotal'].mean(),r2['ctotal'].mean(),r3['ctotal'].mean() ,r4['ctotal'].mean()]
ri = [r1['inctotal'].mean(),r2['inctotal'].mean(),r3['inctotal'].mean() ,r4['inctotal'].mean()]
rw = [r1['wtotal'].mean(),r2['wtotal'].mean(),r3['wtotal'].mean() ,r4['wtotal'].mean()]
df = pd.DataFrame({ "c" : np.array(rc), "inc" :np.array(ri), "w" : np.array(rw), "region" : ['1','2','3','4']}, index = [1, 2, 3, 4])


plt.scatter(df['inc'],df['c'])
plt.title('Mean Income vs Mean Consumption - by Region')

#%% 3.2) Inequality 

r1=r1.fillna(0)
r2=r2.fillna(0)
r3=r3.fillna(0)
r4=r4.fillna(0)

## Gini for consumption

c1 = np.array(r1['ctotal'])
g1c = gini(c1)

c2 = np.array(r2['ctotal'])
g2c = gini(c2)

c3 = np.array(r3['ctotal'])
g3c = gini(c3)

c4 = np.array(r4['ctotal'])
g4c = gini(c4)

del c1,c2,c3,c4

plt.scatter(ri,[g1c,g2c,g3c,g4c])
plt.title('Mean Income vs Gini Consumption - by Region')

## Gini for wealth

c1 = np.array(r1['wtotal'])
g1w = gini(c1)

c2 = np.array(r2['wtotal'])
g2w = gini(c2)

c3 = np.array(r3['wtotal'])
g3w = gini(c3)

c4 = np.array(r4['wtotal'])
g4w = gini(c4)

del c1,c2,c3,c4

plt.scatter(ri,[g1w,g2w,g3w,g4w])
plt.title('Mean Income vs Gini Wealth - by Region')

# Gini for Income

c1 = np.array(r1['inctotal'])
g1i = gini(c1)

c2 = np.array(r2['inctotal'])
g2i = gini(c2)

c3 = np.array(r3['inctotal'])
g3i= gini(c3)

c4 = np.array(r4['inctotal'])
g4i = gini(c4)

del c1,c2,c3,c4


plt.scatter(ri,[g1i,g2i,g3i,g4i])
plt.title('Mean Income vs Gini Income - by Region')

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

                Q2 & Q3 - CIW Analysis & Inequality
                        LABOR SUPPLY

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
