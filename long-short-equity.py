import numpy as np
import pandas as pd
import math
import datetime as datetime
from datetime import date
import re

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

class long_short_preds(object):
    
    def __init__(self, all_stocks, all_data, training_duration, starts_on, for_days):
        
        self.all_stocks = all_stocks
        self.all_data = all_data
        self.opendays_list = [item.date() for item in self.all_data[self.all_stocks[0]].index]+[datetime.date(2020, 8, 5)] #self.all_data[self.all_stocks[0]].index.to_list()
        self.training_duration = training_duration*2
        
        self.starts_on = starts_on
        while self.starts_on not in self.opendays_list:
            self.starts_on = self.starts_on + datetime.timedelta(days=1)
        
        self.for_days = for_days
        self.training_starts_on = self._nearest(self.starts_on - datetime.timedelta(days=self.training_duration*2))
                                           
        self.algo1_rf_preds = {}
        self.test_days = []
        
        self.train_test_split_ratio = 0.65
        self.test_days = self._create_days()                        
                
        self.droprows = 5
        self.testing_ends_on = self.test_days[-1]
                                           
        self.X_train, self.y_train, self.X_test, self.y_test = {}, {}, {}, {}
                                           
        self.dayfeatures = ["Year","Month","Week","Day","Dayofweek","Dayofyear","Is_month_end","Is_month_start",
          "Is_quarter_end","Is_quarter_start","Is_year_end","Is_year_start"]
        
    def _create_days(self):
        day = self.starts_on
        # days=[self.starts_on]
        days=[]
        while self.for_days!=1:
            day= day + datetime.timedelta(days=1)

            if day in self.opendays_list:
                days.append(day)

                if len(days)==self.for_days:
                    break
        # days = pd.Series((v[0] for v in days), dtype=datetime.date)
        # print(type(days))
        if self.for_days==1:
          z=1
          while True:
            tester = self.starts_on + datetime.timedelta(days=z)
            z+=1
            if tester in self.opendays_list:
             #self.starts_on not in self.opendays_list:
            # self.starts_on = self.starts_on + datetime.timedelta(days=1)
              days.append(tester)
              break

        return days
        
    def __repr__(self):
#         pass
#         return f"{self.testing_ends_on}"#, {self.starts_on}, {self.total_days}, {self.rebalance_every}"
        for item in self.final_dfs:
#             print(self.final_dfs[item].isnull().sum())   
#             print(self.X_train[item].isnull().sum())
#             print(self.y_test[item].isnull().sum())
            print(self.algo1_rf_preds[item])

    def _nearest(self, pivot):
        return min(self.opendays_list, key=lambda x: abs(x - pivot))
        
    def preds_for_days(self):
        for stock in self.all_stocks:
            self.algo1_rf_preds[stock] = self._algo_randomforest(stock)            
            
        # index = self.X_test[stock].index
        index=self.test_days
        self.algo1_rf_preds = pd.DataFrame(self.algo1_rf_preds, index=index)

            
        return self.algo1_rf_preds
            
            
    def _algo_randomforest(self, stock): #X_train, y_train, X_test):

        # def randomforest(X_train, ):
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

        # Use the random grid to search for best hyperparameters

        # Random search of parameters, using 3 fold cross validation, 
        # search across 100 different combinations, and use all available cores
        rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                                       n_iter = 100, cv = tscv, verbose=2, random_state=13, 
                                       n_jobs = -1)# Fit the random search model
        
        rf_random.fit(self.X_train[stock], self.y_train[stock])
        # rf_random.best_params_
        best_random = rf_random.best_estimator_

        predictions = best_random.predict(self.X_test[stock])
#         print(predictions)
        return predictions                   
            
    
    def test_train_sets(self):
        for stock,df in self.final_dfs.items():
            (self.X_train[stock], self.y_train[stock]), (self.X_test[stock], self.y_test[stock]) = self._get_train_test(df)
             
    
    def add_features(self):

        for stock,df in self.final_dfs.items():
            df.reset_index(inplace=True)
            df = self._add_datepart(df, "Date", drop=False)
        #     df = get_technical_indicators(df)
            df = pd.get_dummies(df, columns=self.dayfeatures, drop_first=True)
#             df.dropna(axis=0, inplace=True)
#             df.drop(df.tail(1).index,inplace=True)
            df.set_index("Date", inplace=True)
            self.final_dfs[stock] = df        
#         return f"{self.final_dfs}"


    def _get_train_test(self, df):
        data = df.copy()
        y = data[["rtns"]]
        X = data.drop(["rtns"], axis=1)

#         train_samples = int(X.shape[0]-self.for_days-1)
        test_samples = X.shape[0]-int(self.for_days)

        X_train = X.iloc[self.droprows:test_samples]
        X_test = X.iloc[test_samples:]

        y_train = y.iloc[self.droprows:test_samples]
        y_test = y.iloc[test_samples:]

        return (X_train, y_train), (X_test, y_test)
        
    def basic_data_prep(self):
        final_dfs = {}
        drop_cols = ["Symbol", "Series"]#, "Open","High","Low","Last"]

        for stock in self.all_stocks:
            final_dfs[stock] = self.all_data[stock].copy()
            final_dfs[stock].drop(labels=drop_cols, axis=1, inplace=True)
            final_dfs[stock].index = pd.to_datetime(final_dfs[stock].index)
                
            
            final_dfs[stock] = pd.DataFrame(final_dfs[stock][self.training_starts_on:self.test_days[-1]])


            final_dfs[stock]["1_diff"] = final_dfs[stock].Close-final_dfs[stock].Close.shift(1)
            final_dfs[stock]["2_diff"] = final_dfs[stock].Close-final_dfs[stock].Close.shift(2)
            final_dfs[stock]["3_diff"] = final_dfs[stock].Close-final_dfs[stock].Close.shift(3)
            final_dfs[stock]["4_diff"] = final_dfs[stock].Close-final_dfs[stock].Close.shift(4)

            final_dfs[stock]['rtns'] = (final_dfs[stock].Close/final_dfs[stock].Close.shift(1)-1)*100
            final_dfs[stock]['rtns'] = final_dfs[stock]['rtns'].shift(-1)
            
#             final_dfs[stock].dropna(axis=0, inplace=True)
            
        self.final_dfs = final_dfs
#         return final_dfs

    def _add_datepart(self, df, fldname, drop=True, time=False):
        "Helper function that adds columns relevant to a date."
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
    


def daily_trades(list_dict, trade_day):
    Hitrate = 4.0 #Out of 10 trades
    Txns = .25 #Transaction costs + Liquidity costs(20% in excess of loss)
    Max_loss = -250.0  #Per_trade cost limit

    columns = ["Date", "Equity_Symbol","Type", "Entry","Exit","Noshares","Stop_loss","Reward",
               "GrossLoss","MaxProfit","MaxLoss","ExpectedRtns", "PnL"]
    
    trade_data = pd.DataFrame(columns=columns)

    for stock,value in list_dict:
        data = all_data[stock].copy(deep=True)
        # data.index = pd.to_datetime(data.index)
        pre_close = data.loc[trade_day]["Prev Close"]

        if value<0: #(Selling Stocks)       
            new_row = {
                'Date' : trade_day,
                'Equity_Symbol' : stock,
                'Type' : "SELL",
                'Entry' : data.loc[trade_day].Open,            
                'High' : data.loc[trade_day].High,
                'Low' : data.loc[trade_day].Low,
                # 'Stop_loss' : float(data.loc[:trade_day][-2:-1]['High'].values*1.015)
                # 'Stop_loss' : float("{:.2f}".format(data.loc[:trade_day][-2:-1]['High'].values*1.015))
            }
            # new_row['Stop_loss'] = float(data.loc[:trade_day][-1:]['Open'].values*1.015)
            # new_row['Reward']    = pre_close*(100+value)/100
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
                # 'Stop_loss' : float(data.loc[:trade_day][-2:-1]['Low'].values*.985)
                # 'Stop_loss' : float("{:.2f}".format(data.loc[:trade_day][-2:-1]['Low'].values*1.015))
            }
            # new_row['Stop_loss'] = float(data.loc[:trade_day][-1:]['Open'].values*0.985)
            # new_row['Reward']    = pre_close*(100+value)/100
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
            
#             trade_data.set_index("Date", inplace=True)
            
        
    return trade_data
#     return trade_data.set_index("Date")


import glob
import streamlit as st
from nsepy import get_history
import os



root = f"{os.getcwd()}/data/"
classifications = {}
all_stocks = []

for file in glob.glob(root+'*.csv'):
    classifications[file[len(root):-4]] = pd.read_csv(file)
    
classified_list = {}
for item in classifications:
    classified_list[f'{item}'] = classifications[f'{item}']['Symbol'].tolist()

# for item in classified_list:
    # print(item)

# clubs = st.multiselect('Choose among Classifications!', classified_list.keys())
# clubs
st.title("Long-Short-Equity-Pipeline!!")
classifications = [item for item in classified_list]
classifications = st.multiselect('Choose among Classifications!', classifications)

for key in classifications:
    all_stocks = all_stocks + classified_list[key]
    # all_stocks.append(classified_list[key])
all_stocks.append("All")
all_stocks_chosen = st.multiselect('Filter Stocks!', all_stocks)
if "All" in all_stocks_chosen:
    all_stocks.remove("All")
    all_stocks_chosen = all_stocks
    st.write(all_stocks_chosen)
else:
    # all_stocks = all_stocks_chosen.remove("0. All")
    # all_stocks_chosen.remove("0. All")
    st.write(all_stocks_chosen)

# Filter stocks
# print(all_stocks)
@st.cache
def data_update(all_stocks_chosen):
    all_data = {}
    start, end = date(2018,1,1), date(2020,8,4)
    for equity in all_stocks_chosen:
        all_data[equity] = get_history(symbol=equity, start=start, end=end)
    return all_data

all_data = data_update(all_stocks_chosen)
@st.cache
def local_copy(all_stocks_chosen):
    stks = all_stocks_chosen # ["AXISBANK", "ICICIBANK"]
    stks_data = {}

    for stk in stks:
        stks_data[stk] = all_data[stk].copy(deep=True)
        # stks_data[stk].set_index("Date", inplace=True)
        stks_data[stk].index = pd.to_datetime(stks_data[stk].index)

    return stks, stks_data


stks, stks_data = local_copy(all_stocks_chosen)
#[item for item in stks_data.keys()]
check_off = ['<select>']
check_on = st.selectbox('Check out the dataframes!', check_off+all_stocks_chosen)
if check_on!='<select>':
    st.write(stks_data[check_on])
# for item in clubs:
#     st.write(stks_data[item])

list_of_pipelines = ["Our Algo", "Nothing Yet"]

option = st.selectbox(
    'Which algorithm do you like to run?',
     check_off+list_of_pipelines)
     

if option =="Our Algo":
    training_duration = st.number_input('Training Duration')
    training_end = st.date_input(label='Training Ends On')
    results_for = st.number_input(label='Results for how many days')
    rebalance_every = st.number_input(label='Rebalance every?')


    training_duration =  int(training_duration)            # at least 90 days in general
    test_ref_on = training_end   #date(2020,5,29)          # testing starts on
    final_rls = int(results_for)                 # Results for y days
    total_days = int(rebalance_every)                # rebalance every x days (should be less than final_rls(y))
                        

    riches = pd.DataFrame()
    remaining_days = final_rls

    for item in range(math.ceil(final_rls/total_days)):
    # print(item)
        instance2 = long_short_preds(stks, stks_data, training_duration, test_ref_on, total_days)
        test_ref_on = instance2.test_days[-1]
        remaining_days = remaining_days-total_days
        if remaining_days-total_days<=0:
            total_days = remaining_days
        instance2.basic_data_prep()
        instance2.add_features()
        instance2.test_train_sets()
        instance_results = instance2.preds_for_days()

        riches = pd.concat([riches, instance_results], axis=0)

    st.write(riches)

    bitches = pd.DataFrame()
    for _, item in riches.iterrows():
        sdate = _#.date()
        stuff = [tuple(x) for x in zip(item.index, item.to_list())]
        # print(sdate)
        # print(_.date(), stuff)
        dailybitch = daily_trades(stuff, sdate)
        bitches = pd.concat([bitches, dailybitch], axis=0)

    st.write(bitches)
    


