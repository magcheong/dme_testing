import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

import sklearn
import warnings

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import pickle

warnings.filterwarnings('ignore')

# read clinical excel sheet

df = pd.read_excel('./AI DME Data Extraction Sheet Deidentified.xlsx') 

# reduce to Age (Years), Sex (1=Male, 2=Female), Presenting Best Corrected Visual Acuity (Snellen), 
# Phakic status (1=phakic, 2=pseudophakic), 
# Past medical history (0=Nil, 1=Diabetes, 2=Hypertension, 3=Hyperlipidaemia, 4=Smoking),
# Latest HbA1c at date of presentation (%), Post-treatment Best Corrected Visual Acuity (Snellen)

wanted_col = ['Age (Years)', 'Sex (1=Male, 2=Female)', 'Presenting Best Corrected Visual Acuity (Snellen)', 
'Phakic status (1=phakic, 2=pseudophakic)', 
'Past medical history (0=Nil, 1=Diabetes, 2=Hypertension, 3=Hyperlipidaemia, 4=Smoking)',
'Latest HbA1c at date of presentation (%)', 'Post-treatment Best Corrected Visual Acuity (Snellen)']

df = df[wanted_col]

df.rename({'Age (Years)':'age', 'Sex (1=Male, 2=Female)':'sex', 'Presenting Best Corrected Visual Acuity (Snellen)':'pre_va', 
'Phakic status (1=phakic, 2=pseudophakic)':'phakic_status', 
'Past medical history (0=Nil, 1=Diabetes, 2=Hypertension, 3=Hyperlipidaemia, 4=Smoking)':'past_medical_history',
'Latest HbA1c at date of presentation (%)':'pre_HbA1c', 'Post-treatment Best Corrected Visual Acuity (Snellen)':'post_va'}, 
          axis=1, inplace = True)


df['sex'].replace({1: 0}, inplace = True) 
df['sex'].replace({2: 1}, inplace = True) 

df['pre_HbA1c'].replace({'NIL': np.nan}, inplace = True)
df['pre_HbA1c'].fillna(np.mean(df['pre_HbA1c']), inplace = True)

# convert va into logmar

df['pre_logmar_letter'] = ''
df['post_logmar_letter'] = ''

df['pre_letter'] = df['pre_va'].apply(lambda x: x[-2:] if (x[-2] == '-' or x[-2] == '+') else 0)
df['pre_letter'] = df['pre_letter'].astype(int)
df['pre_va_trunc'] = df['pre_va'].apply(lambda x: x[:-2] if (x[-2] == '-' or x[-2] == '+') else x)

df['post_letter'] = df['post_va'].apply(lambda x: x[-2:] if (x[-2] == '-' or x[-2] == '+') else 0)
df['post_letter'] = df['post_letter'].astype(int)
df['post_va_trunc'] = df['post_va'].apply(lambda x: x[:-2] if (x[-2] == '-' or x[-2] == '+') else x)

for i in range(len(df)): 
    if df['pre_va_trunc'].iloc[i] == 'CF 1m':
        df['pre_logmar_letter'].iloc[i] = 1.8
        df['pre_va_trunc'].iloc[i] = '1/1'
    elif df['pre_va_trunc'].iloc[i] == 'CF 2m':
        df['pre_logmar_letter'].iloc[i] = 2
        df['pre_va_trunc'].iloc[i] = '1/1'
    else:
        df['pre_logmar_letter'].iloc[i] = 0
        
    if df['post_va_trunc'].iloc[i] == 'CF 1m':
        df['post_logmar_letter'].iloc[i] = 1.8
        df['post_va_trunc'].iloc[i] = '1/1'
    elif df['post_va_trunc'].iloc[i] == 'CF 2m':
        df['post_logmar_letter'].iloc[i] = 2
        df['post_va_trunc'].iloc[i] = '1/1'
    else:
        df['post_logmar_letter'].iloc[i] = 0        

        #         df['logmar'].loc[i] = df['pre_va_trunc'].iloc[i].apply(lambda x: 1.8 if x[2:].strip() == 1 else 2)
for i in range(len(df)):  
        df['pre_va_trunc_split'] = df['pre_va_trunc'].apply(lambda x: re.split("/", x))
        df['pre_va_numerator'] = df['pre_va_trunc_split'].apply(lambda x: x[0])
        df['pre_va_numerator'] = df['pre_va_numerator'].astype(float)
        df['pre_va_denominator'] = df['pre_va_trunc_split'].apply(lambda x: x[1])
        df['pre_va_denominator'] = df['pre_va_denominator'].astype(float)  

        df['pre_logmar'] = np.log10(df['pre_va_denominator']/df['pre_va_numerator']) - 0.02*df['pre_letter'] + df['pre_logmar_letter']
        df['pre_logmar'] = df['pre_logmar'].apply(lambda x: round(x, 2))
        
        df['post_va_trunc_split'] = df['post_va_trunc'].apply(lambda x: re.split("/", x))
        df['post_va_numerator'] = df['post_va_trunc_split'].apply(lambda x: x[0])
        df['post_va_numerator'] = df['post_va_numerator'].astype(float)
        df['post_va_denominator'] = df['post_va_trunc_split'].apply(lambda x: x[1])
        df['post_va_denominator'] = df['post_va_denominator'].astype(float)  

        df['post_logmar'] = np.log10(df['post_va_denominator']/df['post_va_numerator']) - 0.02*df['post_letter'] + df['post_logmar_letter']
        df['post_logmar'] = df['post_logmar'].apply(lambda x: round(x, 2))


df['diff_logmar'] = df['post_logmar'] - df['pre_logmar']

df1 = df[['age','sex', 'phakic_status','past_medical_history','pre_HbA1c','diff_logmar']]

df1['Diabetes'] = 0
df1['Hypertension'] = 0
df1['Hyperlipidaemia'] = 0
df1['Smoking'] = 0

for i in range(len(df1)):    
    if re.search(r'\b1\b', str(df1['past_medical_history'].loc[i])):
        df1['Diabetes'].iloc[i] = 1
    if re.search(r'\b2\b', str(df1['past_medical_history'].loc[i])):
        df1['Hypertension'].iloc[i] = 1
    if re.search(r'\b3\b', str(df1['past_medical_history'].loc[i])):
        df1['Hyperlipidaemia'].iloc[i] = 1
    if re.search(r'\b4\b', str(df1['past_medical_history'].loc[i])):
        df1['Smoking'].iloc[i] = 1        

df_final = df1.copy()
df_final = df_final[['age', 'sex', 'phakic_status', 'pre_HbA1c',
                    'Diabetes', 'Hypertension', 'Hyperlipidaemia', 'Smoking', 'diff_logmar']]

# RF

X = df_final.iloc[:, :-1]   
y = df_final.iloc[:, -1] 

# get train data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)

RF_regressor = RandomForestRegressor(n_estimators = 250, max_depth = None, random_state = 42)

#Train SRF
RF_regressor.fit(X_train, y_train)
#Save model
filename = open('RF_regressor_clinical.pkl', 'wb') 
# source, destination 
pickle.dump(RF_regressor, filename)  

prediction = RF_regressor.predict(X_test)
mse = mean_squared_error(y_test, prediction)
rmse = mse**.5
r2 = r2_score(y_test, prediction)
print(f'mse: {mse:.3f}')
print(f'rmse: {rmse:.3f}')
print(f'r2: {r2:.3f}')

scores = cross_val_score(RF_regressor, X, y, cv=5, scoring='r2')

r2_mean = np.mean(scores)
r2_std = np.std(scores)

print(f'cross val score: {scores}')
print(f'r2 mean: {r2_mean:.3f}')
print(f'r2 std: {r2_std:.3f}')