# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 17:22:38 2023

@author: andre
"""

#######################################################
#######################################################

#  BOND ANALYSIS: STREAMLIT

#######################################################
#######################################################

#### IMPORT PACKAGES
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import matplotlib.dates as mdates
from PIL import Image
sns.set_theme()
import statsmodels.api as sm
import statsmodels.formula.api as smf
####



# TITLE
st.title('Financial Data Analysis, UK Budget Speeches')

st.markdown("##")
st.markdown("##")

reg = pd.read_csv(r"D:\directory\bond_regression_df.csv")
reg = reg.loc[reg["diff_days"] > -16]

days = reg.groupby("diff_days", as_index=False)[['UK_1_Yr_Govt_Bond_Yield_percent', 'UK_10_Yr_Govt_Bond_Yield_percent',
       'UK_3_Yr_Govt_Bond_Yield_percent', 'FTSE_All_Share',
       'USD_GBP_UK_Spot_Exchange_Rate', 'Negative', 'Neutral', 'Positive',
       'Compound', 'negative_occurences', 'positive_occurences',
       'uncertainty_occurences', 'SP_compound', 'SP_uncertainty',
       'SP_pos_index', 'share_B']].mean()

####

st.markdown("<h2 style='text-align: center; color: blue;'>Charting variables before and after speech</h2>",
            unsafe_allow_html=True)

st.markdown("##")
st.markdown("##")

## PLOT 1: AVERAGE COMPOUND SCORE

fig, ax = plt.subplots()

ax.plot(days['diff_days'], days['Compound'],
        label='UK 1 yr bond', marker='o', markersize=4, color="navy")
ax.axvline(x=0, color='r', alpha=0.5, linestyle='--', linewidth=2)
label_position_y = 0.06
#ax.text(0, label_position_y, 'Day of speech',
 #       fontsize=12, color='black', ha='center')

ax.set_xlabel('Days before and after speech')
ax.set_ylabel('Average compound score of articles')

st.markdown("<h2 style='text-align: center; font-size: 30px;'>Compound score</h2>",
            unsafe_allow_html=True)

st.pyplot(fig)

####

## PLOT 2: SPOT EXCHANGE RATE - LEVELS

fig, ax = plt.subplots()

ax.plot(days['diff_days'], days['USD_GBP_UK_Spot_Exchange_Rate'],
        label='UK 1 yr bond', marker='o', markersize=4, color="navy")
ax.axvline(x=0, color='r', alpha=0.5, linestyle='--', linewidth=2)
label_position_y = 0.06
#ax.text(0, label_position_y, 'Day of speech',
 #       fontsize=12, color='black', ha='center')

ax.set_xlabel('Days before and after speech')
ax.set_ylabel('GBP_USD')


st.markdown("##")
st.markdown("##")
st.markdown("<h2 style='text-align: center; font-size: 30px;'>USD spot exchange rate</h2>",
            unsafe_allow_html=True)


st.pyplot(fig)


## PLOT 3: FTSE All Share

fig, ax = plt.subplots()

ax.plot(days['diff_days'], days['FTSE_All_Share'],
        label='UK 1 yr bond', marker='o', markersize=4, color="navy")
ax.axvline(x=0, color='r', alpha=0.5, linestyle='--', linewidth=2)
label_position_y = 0.06
#ax.text(0, label_position_y, 'Day of speech',
 #       fontsize=12, color='black', ha='center')

ax.set_xlabel('Days before and after speech')
ax.set_ylabel('GBP_USD')


st.markdown("##")
st.markdown("##")
st.markdown("<h2 style='text-align: center; font-size: 30px;'>FTSE All Share</h2>",
            unsafe_allow_html=True)


st.pyplot(fig)

st.markdown("##")
st.markdown("##")



####

st.markdown("<h2 style='text-align: center; color: blue;'>Regression results</h2>",
            unsafe_allow_html=True)
st.markdown("##")
st.markdown("##")

formula = 'UK_1_Yr_Govt_Bond_Yield_percent ~ speech_day_dummy + uncertainty_occurences +  SP_pos_index + SP_uncertainty'

# Fit the OLS model with robust standard errors
model = smf.ols(formula, data=reg).fit(cov_type='HC1')

st.markdown('**Regression of UK 1 year gilt on speech day and sentiment variables**')
table1_html = model.summary().tables[1].as_html()
st.markdown(f"<pre>{table1_html}</pre>", unsafe_allow_html=True)


formula = 'USD_GBP_UK_Spot_Exchange_Rate ~ speech_day_dummy + uncertainty_occurences +  SP_pos_index + SP_uncertainty'

# Fit the OLS model with robust standard errors
model = smf.ols(formula, data=reg).fit(cov_type='HC1')

st.markdown('**Regression of GBP USD spot rate on speech day and sentiment variables**')
table1_html = model.summary().tables[1].as_html()
st.markdown(f"<pre>{table1_html}</pre>", unsafe_allow_html=True)


formula = 'FTSE_All_Share ~ speech_day_dummy + uncertainty_occurences +  SP_pos_index + SP_uncertainty'

# Fit the OLS model with robust standard errors
model = smf.ols(formula, data=reg).fit(cov_type='HC1')

st.markdown('**Regression of FTSE All Share on speech day and sentiment variables**')
table1_html = model.summary().tables[1].as_html()
st.markdown(f"<pre>{table1_html}</pre>", unsafe_allow_html=True)








# streamlit run C:\Users\andre\OneDrive\Documents\bond_presentation.py













