import os

import numpy as np
from django.http import HttpResponse

import json

from django.template import loader
from sklearn.model_selection import train_test_split

from covidDashboard.settings import BASE_DIR

import pandas as pd
from pymongo import MongoClient
import plotly.express as px
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

COVID_CSV_DATA_URL = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"


class CovidDataAutomation:

    def __init__(self, url):
        self.url = url
        self.required_columns = ['location', 'date', 'total_cases_per_million', 'total_deaths_per_million']
        self.allowed_countries = ["India", "China", "United States", "European Union"]

    def run(self):
        print("========================================")
        print("Loading Data...")
        print("----------------------------------------")
        self.load_data()
        print("Starting to process data...")
        self.process_incrementally()
        print("----------------------------------------")
        print("Writing output to json file...")
        self.write_json_file()
        print("========================================")
        print("Processed {} Covid Data Records successfully".format(self.total_dataset_size))
        print("")
        print("Execution Completed.")

    @staticmethod
    def get_start_of_century():
        return '2000-01-01'

    @staticmethod
    def read_from_local_covid_data_file():
        with open('last_fetched_date.txt', 'r') as file:
            lines = file.readline().rstrip()
            yield lines
            return
            """for line in lines:
                yield line
                return"""

    @staticmethod
    def get_last_date_fetched():
        last_fetched_date = "2000-01-01"
        try:
            last_fetched_date = [x for x in CovidDataAutomation.read_from_local_covid_data_file()][0]
            if not last_fetched_date or len(last_fetched_date.strip()) == 0:
                last_fetched_date = CovidDataAutomation.get_start_of_century()

        finally:
            print("----------------------------------------")
            print("Processing records after {}".format(last_fetched_date))
            return last_fetched_date

    @staticmethod
    def store_last_date_fetched(last_fetched_date):
        with open("last_fetched_date.txt", "w") as file:
            file.write(last_fetched_date)

    @staticmethod
    def get_column_name_mappings():
        return {
            'location': 'country',
            'total_cases_per_million': 'total_cases',
            'total_deaths_per_million': 'total_deaths'
        }

    def load_data(self):
        self.data = pd.read_csv(self.url)

    def write_json_file(self):
        records = self.data.to_json(orient='records')
        records = json.loads(records)

        with open('covid_data.json', 'a') as file:
            for record in records:
                file.write(json.dumps(record))

    def process_incrementally(self):
        self.data = self.data[self.required_columns] \
            .rename(columns=CovidDataAutomation.get_column_name_mappings()) \
            .fillna(0)
        self.data = self.data[(self.data["country"].isin(self.allowed_countries))] \
            .sort_values(by='date', ascending=False)

        last_fetched_date = CovidDataAutomation.get_last_date_fetched()
        CovidDataAutomation.store_last_date_fetched(self.data["date"].values[0])

        print("----------------------------------------")
        print("Fetching the last fetched date...")
        self.data = self.data[self.data['date'] > last_fetched_date]
        self.data['id'] = self.data['country'] + "#" + self.data['date']
        self.total_dataset_size = len(self.data.index)

class CausalityModel:

    def __init__(self):
        automation_agent = CovidDataAutomation(COVID_CSV_DATA_URL)
        automation_agent.run()
        try:
            conn = MongoClient()
            print("Connected successfully!!!")
        except:
            print("Could not connect to MongoDB")
        # database
        db = conn.database
        f = open(os.path.join(BASE_DIR, 'covid_data.json'))
        self.collection = db.covid_cases
        self.countries = set()
        lines = f.readlines()
        for line in lines:
            i = json.loads(line)
            if self.collection.find({'_id':i['id']}).count() == 0:
                new_record = {
                    "_id":i['id'],
                    "date":i['date'],
                    "country":i['country'],
                    "cases":i['total_cases'],
                    "deaths":i['total_deaths'],
                }
                rec =  self.collection.insert_one(new_record)
                print("Data inserted with record ids",rec)
            self.countries.add(i['country'])

        self.country_dataFrames = dict()
        for country in self.countries:
            country_data = self.collection.find({'country':country})
            self.country_dataFrames[country] = pd.DataFrame(country_data)

        co = 0
        for country in self.country_dataFrames:
            temp = self.country_dataFrames[country]
            if co == 0:
                temp['cases'] = temp['cases'].astype(float)
                df_cases = temp[['date', 'cases']].rename(columns={'cases':country})
                temp['deaths'] = temp['deaths'].astype(float)
                df_deaths = temp[['date', 'deaths']].rename(columns={'deaths':country})
                co = 1
            else:
                temp['cases'] = temp['cases'].astype(float)
                temp1 = temp[['date', 'cases']]
                df_cases = df_cases.merge(temp1, on='date', how='right').rename(columns={'cases':country})
                temp['deaths'] = temp['deaths'].astype(float)
                temp1 = temp[['date', 'deaths']]
                df_deaths = df_deaths.merge(temp1, on='date', how='right').rename(columns={'deaths':country})

        df_cases = df_cases.dropna()
        df_cases['date'] =  pd.to_datetime(df_cases['date'])
        self.df_cases = df_cases.set_index('date').rename_axis('country', axis=1)
        df_deaths = df_deaths.dropna()
        df_deaths['date'] =  pd.to_datetime(df_deaths['date'])
        self.df_deaths = df_deaths.set_index('date').rename_axis('country', axis=1)

        self.country_dataFrames_diff = dict()
        for country in self.countries:
            country_data = list(self.collection.find({'country':country}))
            country_data = sorted(country_data, key = lambda i: (i['date']), reverse = True)
            for i in range(0,len(country_data)-1):
                diff = max(0,country_data[i]['cases'] - country_data[i+1]['cases'])
                country_data[i]['cases'] = diff
                diff = max(0,country_data[i]['deaths'] - country_data[i+1]['deaths'])
                country_data[i]['deaths'] = diff
            self.country_dataFrames_diff[country] = pd.DataFrame(country_data)

        co = 0
        for country in self.country_dataFrames_diff:
            temp = self.country_dataFrames_diff[country]
            if co == 0:
                temp['cases'] = temp['cases'].astype(float)
                df_cases = temp[['date', 'cases']].rename(columns={'cases':country})
                temp['deaths'] = temp['deaths'].astype(float)
                df_deaths = temp[['date', 'deaths']].rename(columns={'deaths':country})
                co = 1
            else:
                temp['cases'] = temp['cases'].astype(float)
                temp1 = temp[['date', 'cases']]
                df_cases = df_cases.merge(temp1, on='date', how='right').rename(columns={'cases':country})
                temp['deaths'] = temp['deaths'].astype(float)
                temp1 = temp[['date', 'deaths']]
                df_deaths = df_deaths.merge(temp1, on='date', how='right').rename(columns={'deaths':country})
        df_cases = df_cases.dropna()
        df_cases['date'] = pd.to_datetime(df_cases['date'])
        df_cases = df_cases.sort_values(by='date')
        self.df_cases_diff = df_cases.set_index('date').rename_axis('country', axis=1)
        self.df_train_transformed = self.df_cases_diff.diff().dropna()

        df_deaths = df_deaths.dropna()
        df_deaths['date'] =  pd.to_datetime(df_deaths['date'])
        self.df_deaths_deaths = df_deaths.set_index('date').rename_axis('country', axis=1)
        self.df_train_transformed_deaths = self.df_deaths_deaths.diff().dropna()


    def visualizeCases(self, df):
        fig = px.line(df, facet_col="country", title='Cases Graph', facet_col_wrap=1)
        fig.update_yaxes(matches=None)
        graph = fig.to_html(full_html=False, default_height=500, default_width=700)
        return graph

    def visualizeCasesArea(self, df):
        fig = px.area(df, facet_col="country", title='Cases Area Graph', facet_col_wrap=1)
        fig.update_yaxes(matches=None)
        graph = fig.to_html(full_html=False, default_height=500, default_width=700)
        return graph

    def visualizeDeaths(self, df):
        fig = px.line(df, facet_col="country", title='Deaths Graph', facet_col_wrap=1)
        fig.update_yaxes(matches=None)
        graph = fig.to_html(full_html=False, default_height=500, default_width=700)
        return graph

    def visualizeDeathsArea(self, df):
        fig = px.area(df, facet_col="country", title='Deaths Area Graph', facet_col_wrap=1)
        fig.update_yaxes(matches=None)
        graph = fig.to_html(full_html=False, default_height=500, default_width=700)
        return graph

    def adftest(self, df):
        def adf_test(df):
            result = adfuller(df.values)
            print('ADF Statistics: %f' % result[0])
            print('p-value: %f' % result[1])
            print('Critical values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))

        n_obs = 20
        df_train, df_test = df[0:-n_obs], df[-n_obs:]
        for k in self.country_dataFrames:
            print('ADF Test: ' + k + ' time series')
            adf_test(df_train[k])

    def kpsstest(self, df):
        def kpss_test(df):
            statistic, p_value, n_lags, critical_values = kpss(df.values)

            print(f'KPSS Statistic: {statistic}')
            print(f'p-value: {p_value}')
            print(f'num lags: {n_lags}')
            print('Critial Values:')
            for key, value in critical_values.items():
                print(f'   {key} : {value}')

        n_obs = 20
        df_train, df_test = df[0:-n_obs], df[-n_obs:]
        for k in self.country_dataFrames:
            print('KPSS Test: ' + k + ' time series')
            kpss_test(df_train[k])

    def vartest(self, df):
        model = VAR(df)
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

        for col, val in zip(df.columns, out):
            print(col, ':', round(val, 2))
        n_obs = 20
        df_train, df_test = df[0:-n_obs], df[-n_obs:]
        lag_order = results.k_ar
        df_input = df.values[-lag_order:]
        df_forecast = results.forecast(y=df_input, steps=n_obs)
        df_forecast = (pd.DataFrame(df_forecast, index=df_test.index, columns=df_test.columns + '_pred'))

        def invert_transformation(df, pred):
            forecast = df_forecast.copy()
            columns = df.columns
            for col in columns:
                forecast[str(col)+'_pred'] = df[col].iloc[-1] + forecast[str(col)+'_pred'].cumsum()
            return forecast
        output = invert_transformation(df_train, df_forecast)

        combined = pd.concat([output['India_pred'], df_test['India'], output['European Union_pred'], df_test['European Union'], output['United States_pred'], df_test['United States'], output['China_pred'], df_test['China']], axis=1)
        for k in self.country_dataFrames:
            s_pred = k + '_pred'
            rmse = mean_squared_error(combined[s_pred], combined[k], squared=False)
            mae = mean_absolute_error(combined[s_pred], combined[k])
            print('Forecast accuracy of ' + k)
            print('RMSE: ', round(rmse,2))
            print('MAE: ', round(mae,2))

    def granger_causality(self, df):
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
        print(grangers_causation_matrix(df, variables = df.columns))

    def prediction_regression(self, df, country1, country2):
        country_to_index = dict({df.columns[0]:0,df.columns[1]:1,df.columns[2]:2,df.columns[3]:3})
        cases = df.to_numpy()
        cases_country1 = cases[:,country_to_index[country1]]
        cases_country2 = cases[:,country_to_index[country2]]

        X, X_test, y, y_test = train_test_split(cases_country1, cases_country2, test_size=0.25, random_state=42)
        X = X.reshape(-1,1)
        X_test = X_test.reshape(-1,1)
        y = y.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        regr = RandomForestRegressor(max_depth=2, random_state=0)
        regr.fit(X, y)
        output = regr.predict(X_test)

        output = output.reshape(-1,1)
        to_plot = np.concatenate((output, y_test), axis=1)
        df_plots = pd.DataFrame(to_plot, columns=[country2+' predicted', country2])
        fig = px.line(df_plots, title='Prediction between ' + country1 + ' and ' + country2, facet_col_wrap=1)
        fig.update_yaxes(matches=None)
        graph = fig.to_html(full_html=False, default_height=500, default_width=700)
        return graph

def index(request):
    causalityModel = CausalityModel()
    context = {}
    #context['graphCases'] = causalityModel.visualizeCases(causalityModel.df_cases)
    context['graphCasesArea'] = causalityModel.visualizeCasesArea(causalityModel.df_cases)
    context['graphCasesDaily'] = causalityModel.visualizeCases(causalityModel.df_cases_diff)
    context['graphCasesAreaDaily'] = causalityModel.visualizeCasesArea(causalityModel.df_cases_diff)
    context['graphAdfTest'] = causalityModel.visualizeCasesArea(causalityModel.df_train_transformed)
    causalityModel.adftest(causalityModel.df_cases_diff)
    causalityModel.adftest(causalityModel.df_train_transformed)
    causalityModel.kpsstest(causalityModel.df_cases_diff)
    causalityModel.kpsstest(causalityModel.df_train_transformed)
    causalityModel.vartest(causalityModel.df_cases_diff)
    causalityModel.vartest(causalityModel.df_train_transformed)
    causalityModel.granger_causality(causalityModel.df_cases_diff)
    causalityModel.granger_causality(causalityModel.df_train_transformed)
    countries_list = list(causalityModel.countries)
    for i in range(len(countries_list)):
        for j in range(i):
            a = countries_list[i]
            b = countries_list[j]
            if a == 'United States':
                a = 'UnitedStates'
            if a == 'European Union':
                a = 'European'
            if b == 'United States':
                b = 'UnitedStates'
            if b == 'European Union':
                b = 'European'
            context['graphPrediction'+a+b] =causalityModel.prediction_regression(causalityModel.df_cases_diff, countries_list[i], countries_list[j])
            context['graphPrediction'+b+a] =causalityModel.prediction_regression(causalityModel.df_cases_diff, countries_list[j], countries_list[i])
    #context['graphDeaths'] = causalityModel.visualizeDeaths(causalityModel.df_deaths)
    context['graphDeathsArea'] = causalityModel.visualizeDeathsArea(causalityModel.df_deaths)
    context['graphDeathsDaily'] = causalityModel.visualizeDeaths(causalityModel.df_deaths_deaths)
    context['graphDeathsAreaDaily'] = causalityModel.visualizeDeathsArea(causalityModel.df_deaths_deaths)
    context['graphAdfDeathsTest'] = causalityModel.visualizeDeathsArea(causalityModel.df_train_transformed_deaths)
    causalityModel.adftest(causalityModel.df_deaths_deaths)
    causalityModel.adftest(causalityModel.df_train_transformed_deaths)
    causalityModel.kpsstest(causalityModel.df_deaths_deaths)
    causalityModel.kpsstest(causalityModel.df_train_transformed_deaths)
    causalityModel.vartest(causalityModel.df_deaths_deaths)
    causalityModel.vartest(causalityModel.df_train_transformed_deaths)
    causalityModel.granger_causality(causalityModel.df_deaths_deaths)
    causalityModel.granger_causality(causalityModel.df_train_transformed_deaths)
    countries_list = list(causalityModel.countries)
    for i in range(len(countries_list)):
        for j in range(i):
            a = countries_list[i]
            b = countries_list[j]
            if a == 'United States':
                a = 'UnitedStates'
            if a == 'European Union':
                a = 'European'
            if b == 'United States':
                b = 'UnitedStates'
            if b == 'European Union':
                b = 'European'
            context['graphDeathsPrediction'+a+b] =causalityModel.prediction_regression(causalityModel.df_deaths_deaths, countries_list[i], countries_list[j])
            context['graphDeathsPrediction'+b+a] =causalityModel.prediction_regression(causalityModel.df_deaths_deaths, countries_list[j], countries_list[i])
    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))