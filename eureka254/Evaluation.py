from gluonts.evaluation import Evaluator
import os
import numpy as np
import pandas as pd
from sklearn.metrics import *
from eureka254.DB import DB
from ipywidgets import *
import matplotlib.pyplot as plt


DB = DB()

class Evaluation:
  
    def __init__(self):

        self.odds_df = DB.get_all_odds()

    # evaluate using gluons internal metrics
    def gluonts_evaluation(self, tss, preds, load_path, test_ds):
  
        evaluator = Evaluator(quantiles = [0.1, 0.3, 0.5, 0.7, 0.9])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(preds), num_series=len(test_ds))
        item_metrics.to_csv(os.path.join(load_path, 'models_evaluation_metric.csv'))
        return item_metrics

    # custom profit metric evaluation
    # still requires validation
    def profit_evaluation(self, test_ds, forecasts, teams_df, arrays=False):
  
        def calc_profit(preds_df):
            preds_df['investment'] = np.nan
            preds_df['profit'] = np.nan
            for index in preds_df.index.values:

                odd = preds_df.at[index, 'odds']
                pred = preds_df.at[index, 'preds_over25']
                true = preds_df.at[index, 'test_over25']

                # set investment to KES 50 if the prediction was to place the bet
                if pred == 1:
                    investment = 50
                else:
                    investment = 0
                # if the model got the prediction right, calculate the profit
                if pred == 1 and true == 1:
                    profit = investment * odd - investment
                # if the model got the prediction wrong, deduct the loss
                elif pred == 1 and true == 0:
                    profit = -50
                # else the model predicted a "no bet"
                else:
                    profit = 0

                preds_df.at[index, 'investment'] = investment
                preds_df.at[index, 'profit'] = profit

            return preds_df

        prediction_length = forecasts[0].mean.shape[0]

        odds_dict = {
        'over15_odds':1.5,
        'over25_odds':2.5,
        'over35_odds':3.5,
        'over45_odds':4.5,
        }

        r2s = pd.DataFrame(index=np.arange(0, len(forecasts)), columns=['r2', 'mse', 'acc', 'profit','profit_rate', 'odd_name', 'investment'])

        for i in range(len(forecasts)):

            if not arrays:

                y_test_ = list(test_ds)[i]['target']
                y_preds_ = forecasts[i].mean
                y_test_ = y_test_[-prediction_length:]
                
            else:
                
                y_test_ = test_ds.iloc[i]
                y_preds_ = forecasts.iloc[i]


            r2_score_ = r2_score(y_test_, y_preds_)
            mse_ = mean_squared_error(y_test_, y_preds_)
            mae_ = mean_absolute_error(y_test_, y_preds_)

            preds_df = pd.DataFrame([y_test_, y_preds_], index=['test', 'pred']).T

            odd_names = ['over15_odds', 'over25_odds', 'over35_odds','over45_odds']

            valid_teams = teams_df.iloc[i, -prediction_length:].values


            # initialize best profit as extremely low
            best_profit = -10000

            preds_df_temp = preds_df.copy(deep=True)

            # calculate the profit for all odds
            for odd_name in odd_names:

                preds_df_temp['test_over25'] = (preds_df_temp['test'] > odds_dict.get(odd_name)).astype(int)
                preds_df_temp['preds_over25'] = (preds_df_temp['pred'] > odds_dict.get(odd_name)).astype(int)
                preds_df_temp['teams'] = valid_teams 

                preds_df_temp_bet = preds_df_temp.loc[preds_df_temp['preds_over25']==1].copy(deep=True)

                

                if preds_df_temp_bet.shape[0] > 0:
                
                    acc_ = accuracy_score(preds_df_temp_bet['test_over25'], preds_df_temp_bet['preds_over25'])
                    preds_df_temp_bet['odds'] = self.odds_df.loc[preds_df_temp_bet.teams, odd_name].values
                    preds_df_temp_bet = calc_profit(preds_df_temp_bet)
                    total_investment = preds_df_temp_bet.investment.sum()
                    profit = preds_df_temp_bet.profit.sum()
                
                else:
                
                    acc_=0
                    total_investment=0
                    profit = 0

                

                if profit > best_profit:
                    best_profit = profit
            #       best_odd = odd
                    best_odd_name = odd_name
                    best_acc = acc_
                    best_total_investment = total_investment




            r2s.at[i, 'r2'] = np.round(r2_score_, 2)
            r2s.at[i, 'mse'] = np.round(mse_ , 2)
            r2s.at[i, 'mae'] = np.round(mae_, 2) 
            r2s.at[i, 'acc'] = np.round(best_acc, 2)
            #   r2s.at[i, 'odd'] = np.round(best_odd, 2) 
            r2s.at[i, 'odd_name'] = best_odd_name
            r2s.at[i, 'profit'] = np.round(best_profit, 2)
            if best_total_investment!=0:
                r2s.at[i, 'profit_rate'] = np.round((best_profit / best_total_investment) * 100, 2)
                r2s.at[i, 'investment'] = np.round(best_total_investment, 2)
            else:
                r2s.at[i, 'profit_rate'] = 0
                r2s.at[i, 'investment'] = 0





                
        return r2s.sort_values(by='profit_rate', ascending=False)

    # this displays a widget for selecting the time series to plot controlled by the 'w' parameter.
    # w=5 is chosen arbitrarily here and will require looking into (make it generic)
    def plot_preds_graph(self, tss, preds, plot_length=300, prediction_intervals = (50.0, 90.0)):

        def update(w = 5):

            legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
            fig, ax = plt.subplots(1, 1, figsize=(10, 7))
            tss[w][-plot_length:].plot(ax=ax)  # plot the time series
            preds[w].plot(prediction_intervals=prediction_intervals, color='g')
            plt.grid(which="both")
            plt.legend(legend, loc="upper left")

            fig.canvas.draw()

        interact(update);
  
  
  
    
