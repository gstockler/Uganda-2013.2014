#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 23:25:44 2019

@author: gabi
"""

# =============================================================================
#  LABOR SUPPLY DATA
# =============================================================================

# info on labor supply --> HH survey

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

#%% LABOR SUPPLY - at HH level

ls = pd.read_stata('GSEC8_1.dta', convert_categoricals=False)

# h8q4: work (yes/no) past year
# h8q16: looking for job
# h8q36a-g: hours per day of the week


ls = ls[["HHID","h8q4","h8q5","h8q16","h8q36a","h8q36b","h8q36c","h8q36d","h8q36e","h8q36f","h8q36g"]]
ls.columns = ["hh","work_week","work_year","search","hsun","hmon","htue","hwed","hthur","hfri","hsat"]
ls["hweek"] = ls[["hsun","hmon","htue","hwed","hthur","hfri","hsat"]].sum(axis=1)
ls=ls.drop(["hsun","hmon","htue","hwed","hthur","hfri","hsat"],axis=1)
ls.describe()

ls["work_week"] = ls["work_week"].replace(2, 0)
ls["work_week"] = ls["work_week"].replace(np.nan, 0)
ls = ls.groupby(by="hh")[["work_week","hweek"]].sum(axis=1)
ls.loc[ls.work_week > 0, "hweek"] = ls['hweek']/ ls['work_week'] # mean hours per week worked by HH
ls.loc[ls.work_week > 0, "hh_work"] = 1
ls["hh_work"] = ls["hh_work"].replace(np.nan, 0)

ls.loc[ls.hh_work > 0, "hh_hour"] = ls['hweek']
ls["hh_hour"] = ls["hh_hour"].replace(np.nan, 0)

ls=ls[['hh_work','hh_hour']]
ls.describe()

# there are HH working more than possible hours per week
# so let the max hours per week be 140 (20h per day)
ls.loc[ls.hh_hour > 140, "hh_hour"] = 140

ls.describe()

work_total = len(ls[ls.hh_work == 1]) # how many HH are working
hours_total = np.array(ls[["hh_hour"]].sum())[0]

ls["hh_hour_prop"] =ls["hh_hour"].divide(hours_total)

ls["hh"] = np.array(ls.index.values)

#%% SAVING 

os.chdir('/Users/gabi/Dropbox/2019.1/Development/PS1/UG_2013_14_GGSB')
ls.to_csv("labor_supply.csv")
