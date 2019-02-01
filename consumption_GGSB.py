#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 16:42:32 2019

@author: gabi
"""

# =============================================================================
#  CONSUMPTION DATA
# =============================================================================

# info on income consumption --> C

import pandas as pd
import numpy as np
import os
dollars = 2586.89

pd.options.display.float_format = '{:,.2f}'.format

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/')
from data_functions_albert import remove_outliers

# os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB') - code directory


# http://microdata.worldbank.org/index.php/catalog/2663/download/38438

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB/UGA_2013_14_data')

#%% FOOD CONSUMPTION

c2 = pd.read_stata('GSEC15B.dta', convert_categoricals=False)
c2 = c2[["HHID","itmcd","h15bq4","h15bq5","h15bq6","h15bq7","h15bq8","h15bq9","h15bq10","h15bq11","h15bq12","h15bq13"]]
c2.columns = ["hh","code", "purch_home_quant","purch_home_value","purch_away_quant","purch_away_value","own_quant","own_value","gift_quant","gift_value", "m_p", "gate_p"]
pricescons = c2.groupby(by="code")[["m_p", "gate_p"]].median()
pricescons.to_csv("pricesfood.csv")

## To get own-produced livestock consumption (and use it from the income side)
livestock = c2.loc[c2["code"].isin([117,118,119,120,121,122,123,124,125]),["hh","own_value"]]
livestock = livestock.groupby(by="hh").sum()*52

livestock.to_csv("c_animal.csv")
suml = livestock.describe()/dollars

#Aggregate across items
c2 = c2.groupby(by="hh")[["purch_home_quant","purch_home_value","purch_away_quant","purch_away_value","own_quant","own_value","gift_quant","gift_value"]].sum()
c2 = c2[["purch_home_value", "purch_away_value", "own_value","gift_value"]]
c2["cfood"] = c2[["purch_home_value", "purch_away_value", "own_value","gift_value"]].sum(axis=1)
c2.rename(columns={'total_value':'cfood'}, inplace=True)
c2.rename(columns={'gift_value':'cfood_gift'}, inplace=True)
c2.rename(columns={'own_value':'cfood_own'}, inplace=True)

c2["cfood_purch"] = c2.loc[:,["purch_home_value","purch_away_value"]].sum(axis=1)
c2["cfood_nogift"] = c2.loc[:,["cfood_purch","own_value"]].sum(axis=1)

# Food consumption at year level
c2 = c2[["cfood", "cfood_nogift", "cfood_own", "cfood_purch", "cfood_gift"]]*52
# Cfood is total value. cfood_nogift is total value minus gifts.
c2.reset_index(inplace=True)

data = c2


#%% NONFOOD NONDURABLE CONSUMPTION

c3 = pd.read_stata('GSEC15C.dta', convert_categoricals=False)
c3 = c3[['HHID','itmcd','h15cq3','h15cq4', 'h15cq5','h15cq6','h15cq7','h15cq8'
               ,'h15cq9','h15cq10']]
c3.columns = ["hh","code","unit","purch_quant","purch_value","own_quant","own_value","gift_quant","gift_value", "m_p"]

#Aggregate across items
c3 = c3.groupby(by="hh")[["purch_quant","purch_value","own_quant","own_value","gift_quant","gift_value"]].sum()

c3['cnodur'] = c3.fillna(0)["purch_value"] + c3.fillna(0)["own_value"] + c3.fillna(0)["gift_value"]
c3["cnodur_nogift"] = c3.loc[:,["purch_value","own_value"]].sum(axis=1)
c3.rename(columns={'gift_value':'cnodur_gift'}, inplace=True)
c3.rename(columns={'own_value':'cnodur_own'}, inplace=True)
c3.rename(columns={'purch_value':'cnodur_purch'}, inplace=True)

# non food non durable consumption at year level
c3 = c3[["cnodur", "cnodur_nogift", "cnodur_own", "cnodur_purch", "cnodur_gift"]]*12
c3.reset_index(inplace=True)

data = data.merge(c3, on="hh", how="outer")


#%% DURABLE CONSUMPTION
c4 = pd.read_stata('GSEC15D.dta', convert_categoricals=False)
c4 = c4[["HHID","h15dq2","h15dq3","h15dq4","h15dq5"]]
c4.columns = ["hh","code","unit","purch_quant","purch_value"]

c4 = c4.groupby(by="hh")[["purch_value"]].sum()

# Durable consumption only asked for purchases
c4.rename(columns={'purch_value':'cdur'}, inplace=True)
c4.reset_index(inplace=True)

data = data.merge(c4, on="hh", how="outer")



#%% Create join variables
data["ctotal"] = data.loc[:,["cfood","cnodur"]].sum(axis=1)
data["ctotal_dur"] = data.loc[:,["cfood","cnodur","cdur"]].sum(axis=1)

data["ctotal_gift"] = data.loc[:,["cfood_gift","cnodur_gift"]].sum(axis=1)
data["ctotal_dur_gift"] = data.loc[:,["ctotal_gift","cdur_gift"]].sum(axis=1)

data["ctotal_nogift"] = data.loc[:,["cfood_nogift","cnodur_nogift"]].sum(axis=1)
data["ctotal_dur_nogift"] = data.loc[:,["cfood_nogift","cnodur_nogift"]].sum(axis=1)

data["ctotal_own"] = data.loc[:,["cfood_own","cnodur_own"]].sum(axis=1)
data["ctotal_dur_own"] = data.loc[:,["ctotal_own","cdur_own"]].sum(axis=1)


cdata_short = data[["hh","ctotal","ctotal_dur","ctotal_gift","ctotal_dur_gift","ctotal_nogift","ctotal_dur_nogift","ctotal_own","ctotal_dur_own","cfood","cnodur","cdur"]]

cdata_short[["ctotal","ctotal_dur","ctotal_gift","ctotal_dur_gift","ctotal_nogift","ctotal_dur_nogift","ctotal_own","ctotal_dur_own","cfood","cnodur","cdur"]]=remove_outliers(cdata_short[["ctotal","ctotal_dur","ctotal_gift","ctotal_dur_gift","ctotal_nogift","ctotal_dur_nogift","ctotal_own","ctotal_dur_own","cfood","cnodur","cdur"]], lq=0.005, hq=0.995 )

cdata_short[["ctotal","ctotal_dur","ctotal_gift","ctotal_dur_gift","ctotal_nogift","ctotal_dur_nogift","ctotal_own","ctotal_dur_own","cfood","cnodur","cdur"]]=cdata_short[["ctotal","ctotal_dur","ctotal_gift","ctotal_dur_gift","ctotal_nogift","ctotal_dur_nogift","ctotal_own","ctotal_dur_own","cfood","cnodur","cdur"]]/dollars
sumc =cdata_short.describe()

print(sumc.to_latex())

#%% SAVING CONSUMPTION FILE

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB')
cdata_short.to_csv("cons.csv", index=False)
