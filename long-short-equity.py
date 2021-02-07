import numpy as np
import pandas as pd
import math
import datetime
from datetime import date
import re
import glob
import streamlit as st
from nsepy import get_history
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("always")

class long_short_preds(object):
    
    def __init__(self, stks, stks_data, training_duration, results_for):
        
        self.all_stocks = stks
        self.all_data = stks_data
        self.for_days = results_for
        self.ends_on = stks_data[stks[0]].index.to_series()[:-self.for_days][-1]
        self.starts_on = stks_data[stks[0]].index.to_series()[:self.ends_on][-training_duration]
        self.forecast_starts = stks_data[stks[0]].index.to_series()[self.ends_on:][1]

        self.algo_rf_preds = {}      
        self.final_dfs = {}
        self.X_train, self.y_train, self.X_test, self.y_test = {}, {}, {}, {}

        self.train_test_split_ratio = 0.65                                           
        self.dayfeatures = ["Year","Month","Week","Day","Dayofweek","Dayofyear","Is_month_end","Is_month_start",
          "Is_quarter_end","Is_quarter_start","Is_year_end","Is_year_start"]

    def basic_data_prep(self):

        drop_cols = ["Symbol", "Series"]
        for stock in self.all_stocks:
            self.final_dfs[stock] = self.all_data[stock].copy()
            self.final_dfs[stock].drop(labels=drop_cols, axis=1, inplace=True)
            self.final_dfs[stock]['Close'] = self.final_dfs[stock]['Close'].shift(-self.for_days)

            self.final_dfs[stock]["1_diff"] = self.final_dfs[stock].Close-self.final_dfs[stock].Close.shift(1)
            self.final_dfs[stock]["2_diff"] = self.final_dfs[stock].Close-self.final_dfs[stock].Close.shift(2)
            self.final_dfs[stock]["3_diff"] = self.final_dfs[stock].Close-self.final_dfs[stock].Close.shift(3)
            self.final_dfs[stock]["4_diff"] = self.final_dfs[stock].Close-self.final_dfs[stock].Close.shift(4)

    def add_features(self):

        for stock,df in self.final_dfs.items():
            df.reset_index(inplace=True)
            df = self._add_datepart(df, "Date", drop=False)
            df = pd.get_dummies(df, columns=self.dayfeatures, drop_first=True)
            df.set_index("Date", inplace=True)
            self.final_dfs[stock] = df        

    def get_test_train_splits(self):
        for stock,df in self.final_dfs.items():
            self.X_train[stock], self.y_train[stock], self.X_test[stock], self.y_test[stock] = self._get_train_test(df)
        # print(self.X_train[stock], self.y_train[stock], self.X_test[stock], self.y_test[stock])

    def final_predictions(self):
        for stock in self.all_stocks:
            self.algo_rf_preds[stock] = self._algo_randomforest(stock)            
            # print(self.algo_rf_preds[stock])
        
        self.algo_rf_preds = pd.DataFrame(self.algo_rf_preds)
        return self.algo_rf_preds

    def _add_datepart(self, df, fldname, drop=True, time=False):
        "Helper function that adds columns relevant to a date - from fastai"

        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64
        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = self.dayfeatures
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)

        return df        

    def _get_train_test(self, df):
        data = df.copy()
        y = data[["Close"]]
        X = data.drop(["Close"], axis=1)

        X_train = X.loc[self.starts_on:self.ends_on]
        y_train = y.loc[self.starts_on:self.ends_on]

        X_test = X.loc[self.forecast_starts:]
        y_test = y.loc[self.forecast_starts:]

        return X_train.ffill(), y_train.ffill(), X_test.ffill(), y_test

    def _algo_randomforest(self, stock):

        rf = RandomForestRegressor(random_state = 13)             # Number of trees in random forest
        n_estimators = [int(x) for x in range(5,100,5)]           # Number of features to consider at every split
        max_features = ['auto', 'sqrt']                           # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(5, 150, num = 5)]  # Minimum number of samples required to split a node
        min_samples_split = [3, 5, 7, 9, 11]                   # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4, 8, 12, 16, 20]            # Method of selecting samples for training each tree
        bootstrap = [True, False]

        tscv = TimeSeriesSplit(n_splits=5)

        # Create the random grid
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}

        # Use the random grid to search for hyper-parameter tuning

        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                       n_iter = 100, cv = tscv, verbose=0, random_state=13, 
                                       n_jobs = -1)# Fit the random search model
        
        self.X_train[stock] = np.nan_to_num(self.X_train[stock])
        self.y_train[stock] = np.nan_to_num(self.y_train[stock])
        self.X_test[stock] = np.nan_to_num(self.X_test[stock])
        rf_random.fit(self.X_train[stock], self.y_train[stock].ravel())
        # rf_random.best_params_
        best_model = rf_random.best_estimator_
        
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # self.X_test[stock] = imp.fit_transform(self.X_test[stock])

        predictions = best_model.predict(self.X_test[stock])
        return predictions                   
    
def random_forest_forecast(start_date, end_date, stks, stks_data):

        training_duration = st.number_input('Train for a duration of ??', min_value=6)
        if training_duration > stks_data[stks[0]].shape[0]:
            st.error('Insufficient Data, choose lesser number of days for training')    

        # training_end = st.date_input('Predictions start on', stks_data[stks[0]].index[-1])
        # if training_end not in stks_data[stks[0]].index:
        #     st.error("Date doesn't exist in the data")

        results_for = st.number_input(label='Forecast for how many days ahead??', min_value=1)
        # rebalance_every = st.number_input(label='Rebalance the model every(days)?', min_value=1)
        # if rebalance_every > training_duration:
        #     st.error(f'Possible rebalancing value is less than {training_duration} days')

        test_instance = long_short_preds(stks, stks_data, training_duration, results_for)
        test_instance.basic_data_prep()
        test_instance.add_features()
        test_instance.get_test_train_splits()
        final_df = test_instance.final_predictions()
        final_df.index = [f'Day_{day+1}' for day in range(results_for)]
        st.markdown('<b><i>Forecasts for all stocks</i></b>', unsafe_allow_html=True)   
        st.write(final_df)

        return final_df, results_for


def basic_template(root):

    classifications = {}
    all_stocks = []

    st.title("Long-Short-Equity-Pipeline!!")

    end_date = datetime.date.today() - datetime.timedelta(days=1)
    start_date = end_date- datetime.timedelta(days=90)

    start_date = st.date_input('Start date', start_date)
    end_date = st.date_input('End date', end_date)
    if start_date < end_date and end_date<datetime.datetime.now().date():
        st.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
    elif end_date>=datetime.datetime.now().date():
        st.error(f'Data only available till yesterday, please select days before {datetime.datetime.now().date()}')
    else:
        st.error('Error: End date must fall after start date.')    

    for file in glob.glob(root+'*.csv'):
        classifications[file[len(root):-4]] = pd.read_csv(file)
        
    classified_list = {}
    for item in classifications:
        classified_list[f'{item}'] = classifications[f'{item}']['Symbol'].tolist()

    classifications = [item for item in classified_list]
    classifications = st.multiselect('Choose among Classifications!', classifications)

    for key in classifications:
        all_stocks = all_stocks + classified_list[key]

    all_stocks.append("All")
    all_stocks_chosen = st.multiselect('Filter Stocks!', all_stocks)
    if "All" in all_stocks_chosen:
        all_stocks.remove("All")
        all_stocks_chosen = all_stocks
        st.write(all_stocks_chosen)
    else:
        st.write(all_stocks_chosen)

    @st.cache
    def data_update(all_stocks_chosen, start_date, end_date):
        all_data = {}
        start, end = start_date, end_date
        for equity in all_stocks_chosen:
            all_data[equity] = get_history(symbol=equity, start=start, end=end)
        return all_stocks_chosen, all_data
    
    stks, stks_data = data_update(all_stocks_chosen, start_date, end_date)

    check_off = ['<select>']
    check_on = st.selectbox('Check out the dataframes!', check_off+all_stocks_chosen)
    if check_on!='<select>':
        st.write(stks_data[check_on])

    list_of_pipelines = ["Random Forest", "Nothing Yet"]

    option = st.selectbox('Which algorithm do you like to run?',
            check_off+list_of_pipelines)    

    return option, start_date, end_date, stks, stks_data

def daily_trades(list_dict, trade_day, stks_data):
    Hitrate = 4.0 #Out of 10 trades
    Txns = .25 #Transaction costs + Liquidity costs(20% in excess of loss)
    Max_loss = -5000  #Per_trade cost limit

    columns = ["Date", "Equity_Symbol","Type", "Entry","Exit","Noshares","Stop_loss","Reward",
               "GrossLoss","MaxProfit","MaxLoss","ExpectedRtns", "PnL"]
    
    trade_data = pd.DataFrame(columns=columns)

    for stock,value in list_dict:
        data = stks_data[stock].copy(deep=True)
        # data.index = pd.to_datetime(data.index)
        pre_close = data.loc[trade_day]["Prev Close"]
        value = value/pre_close-1
        if value<0: #(Selling Stocks)       
            new_row = {
                'Date' : trade_day,
                'Equity_Symbol' : stock,
                'Type' : "SELL",
                'Entry' : data.loc[trade_day].Open,            
                'High' : data.loc[trade_day].High,
                'Low' : data.loc[trade_day].Low,
            }
            new_row['Reward']    = new_row['Entry']*(100+(value*1.618))/100
            new_row['Stop_loss'] = new_row['Entry']*100/(100+value)
            new_row['GrossLoss'] = -(new_row['Stop_loss'] - new_row['Entry']) 
            new_row['MaxProfit'] = -(new_row['Reward'] - new_row['Entry'])
            new_row['MaxLoss']   = new_row['GrossLoss']*(1+Txns)
            new_row['Noshares']  = int(Max_loss/new_row['MaxLoss'])
            new_row['TradeRisk']  = int(new_row['MaxLoss']*new_row['Noshares'])
            new_row['ExpectedRtns']= new_row['MaxProfit']*Hitrate + new_row['MaxLoss']*(10-Hitrate)

            if new_row['Stop_loss'] > data.loc[trade_day].High:
                new_row['Exit'] = data.loc[trade_day].Close
            else:
                new_row['Exit'] = new_row['Stop_loss']

            new_row['PnL'] = -new_row['Noshares']*(new_row['Exit']-new_row['Entry'])

            trade_data = trade_data.append(new_row, ignore_index=True)        

        if value>=0: #(Buying Stocks)
            new_row = {
                'Date' : trade_day,
                'Equity_Symbol' : stock,
                'Type' : "BUY",
                'Entry' : data.loc[trade_day].Open,            
                'High' : data.loc[trade_day].High,
                'Low' : data.loc[trade_day].Low,
            }
            new_row['Reward']    = new_row['Entry']*(100+(value*1.618))/100
            new_row['Stop_loss'] = new_row['Entry']*100/(100+value)
            new_row['GrossLoss'] = new_row['Stop_loss'] - new_row['Entry']
            new_row['MaxProfit'] = new_row['Reward'] - new_row['Entry']
            new_row['MaxLoss']   = new_row['GrossLoss']*(1+Txns)
            new_row['Noshares']  = int(Max_loss/new_row['MaxLoss'])
            new_row['TradeRisk']  = int(new_row['MaxLoss']*new_row['Noshares'])
            new_row['ExpectedRtns']= new_row['MaxProfit']*Hitrate + new_row['MaxLoss']*(10-Hitrate)

            if new_row['Stop_loss'] < data.loc[trade_day].Low:
                new_row['Exit'] = data.loc[trade_day].Close
            else:
                new_row['Exit'] = new_row['Stop_loss']

            new_row['PnL'] = new_row['Noshares']*(new_row['Exit']-new_row['Entry'])

            trade_data = trade_data.append(new_row, ignore_index=True)            
        
    return trade_data


def backtest_results(forecasts_rf, stks_data):
    
    final_results = pd.DataFrame()
    
    for _, item in forecasts_rf.iterrows():
        sdate = _#.date()
        prices = [tuple(x) for x in zip(item.index, item.to_list())]
        daily_trades_list = daily_trades(prices, sdate, stks_data)
        final_results = pd.concat([final_results, daily_trades_list], axis=0)    

    return final_results


def main():
    root = f"{os.getcwd()}/data/"

    option, start_date, end_date, stks, stks_data = basic_template(root)

    if option =="Random Forest":
        forecasts_rf, results_for = random_forest_forecast(start_date, end_date, stks, stks_data)
    elif option=="Nothing Yet":
        st.error('Error: No model exists')

    option = st.selectbox('Would you like a trading backtest?',['<select>', 'Yes', 'No'])

    if option=='Yes':
        forecasts_rf.index = stks_data[stks[0]].index[-results_for:]
        results = backtest_results(forecasts_rf, stks_data)
        st.markdown('<b><i>Backtest Results</i></b>', unsafe_allow_html=True)   
        st.write(results)            


if __name__=='__main__':
    main()