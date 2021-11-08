from pymongo import MongoClient
import json
import plotly.express as px
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


maxlag=15
test = 'ssr_chi2test'

def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

def adf_test(df):
    result = adfuller(df.values)
    print('ADF Statistics: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

def kpss_test(df):
    statistic, p_value, n_lags, critical_values = kpss(df.values)

    print(f'KPSS Statistic: {statistic}')
    print(f'p-value: {p_value}')
    print(f'num lags: {n_lags}')
    print('Critial Values:')
    for key, value in critical_values.items():
        print(f'   {key} : {value}')

try:
    conn = MongoClient()
    print("Connected successfully!!!")
except:
    print("Could not connect to MongoDB")

# database
db = conn.database
f = open('us-states.json')
data = json.load(f)
collection = db.covid_cases
states = set()


for i in data:
    new_record = {
        "_id":i['id'],
        "data":i['Date'],
        "state":i['state'],
        "fips":i['fips'],
        "cases":i['cases'],
        "deaths":i['deaths'],
    }
    states.add(i['state'])

state_dataFrames = dict()
for state1 in states:
    state_data = collection.find({'state':state1})
    if state1 == 'Florida' or state1 == 'Georgia' or state1 == 'Texas':
        state_dataFrames[state1] = pd.DataFrame(state_data)

co = 0
for k in state_dataFrames:
    state = k
    temp = state_dataFrames[state]
    if co == 0:
        temp['cases'] = temp['cases'].astype(int)
        df_cases = temp[['data', 'cases']].rename(columns={'cases':state})
        co = 1
    else:
        temp['cases'] = temp['cases'].astype(int)
        temp1 = temp[['data', 'cases']]
        df_cases = df_cases.merge(temp1, on='data', how='right').rename(columns={'cases':state})

df_cases = df_cases.dropna()
df_cases['data'] =  pd.to_datetime(df_cases['data'])
df_cases = df_cases.set_index('data').rename_axis('state', axis=1)

"""fig = px.line(df_cases, facet_col="state", facet_col_wrap=1, facet_row_spacing=0.008182)
fig.update_yaxes(matches=None)
fig.show()

fig = px.area(df_cases, facet_col='state', facet_col_wrap=1, facet_row_spacing=0.008182)
fig.update_yaxes(matches=None)
fig.show()"""

df_train_transformed = df_cases.diff().dropna()

"""fig = px.line(df_train_transformed, facet_col="state", facet_col_wrap=1)
fig.update_yaxes(matches=None)
fig.show()"""
n_obs = 100

df_test = df_cases[-n_obs:]
df_cases = df_cases[:-n_obs]
for k in state_dataFrames:
    print(k)
    adf_test(df_cases[k])
    kpss_test(df_cases[k])

model = VAR(df_train_transformed)
for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    result = model.fit(i)
    print('Lag Order =', i)
    print('AIC : ', result.aic)
    print('BIC : ', result.bic)
    print('FPE : ', result.fpe)
    print('HQIC: ', result.hqic, '\n')
results = model.fit(maxlags=15, ic='aic')
print(results.summary())

out = durbin_watson(results.resid)

for col, val in zip(df_cases.columns, out):
    print(col, ':', round(val, 2))

ans = grangers_causation_matrix(df_train_transformed, variables = df_train_transformed.columns)
print(ans)

lag_order = results.k_ar
df_input = df_train_transformed.values[-lag_order:]
df_forecast = results.forecast(y=df_input, steps=n_obs)
df_forecast = (pd.DataFrame(df_forecast, index=df_test.index, columns=df_test.columns + '_pred'))

def invert_transformation(df, pred):
    forecast = df_forecast.copy()
    columns = df.columns
    for col in columns:
        forecast[str(col)+'_pred'] = df[col].iloc[-1] + forecast[str(col)+'_pred'].cumsum()
    return forecast
output = invert_transformation(df_cases, df_forecast)
combined = pd.concat([output['Florida_pred'], df_test['Florida'], output['Georgia_pred'], df_test['Georgia'], output['Texas_pred'], df_test['Texas']], axis=1)
print(combined)
rmse = mean_squared_error(combined['Florida_pred'], combined['Florida'], squared=False)
mae = mean_absolute_error(combined['Florida_pred'], combined['Florida'])

print('Forecast accuracy of Florida')
print('RMSE: ', round(rmse,2))
print('MAE: ', round(mae,2))

rmse = mean_squared_error(combined['Georgia_pred'], combined['Georgia'], squared=False)
mae = mean_absolute_error(combined['Georgia_pred'], combined['Georgia'])

print('Forecast accuracy of Georgia')
print('RMSE: ', round(rmse,2))
print('MAE: ', round(mae,2))

rmse = mean_squared_error(combined['Texas_pred'], combined['Texas'], squared=False)
mae = mean_absolute_error(combined['Texas_pred'], combined['Texas'])

print('Forecast accuracy of Texas')
print('RMSE: ', round(rmse,2))
print('MAE: ', round(mae,2))