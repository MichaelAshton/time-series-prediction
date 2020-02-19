### Eureka package usage

import comet_ml

# set comet ml API and REST API Keys here
API_KEY = ""
REST_API_KEY = "" 
workspace= ""

from eureka254.TrainingHarness import TrainingHarness
import os
import numpy as np
from sklearn.metrics import *
from eureka254.DB import DB
from eureka254.ModelKwargs import *
import matplotlib.pyplot as plt

from eureka254.EurekaRegression import EurekaRegression
from eureka254.Evaluation import Evaluation


DB = DB()
TrainingHarness = TrainingHarness()
EurekaRegression = EurekaRegression()
Evaluation = Evaluation()

training_pred_score_path = './'

saved_models_path = os.path.join(training_pred_score_path, 'saved_models')

os.makedirs(saved_models_path, exist_ok=True)

# get golden dataset
df = DB.create_golden_dataset()

# get the teams - teams_id pairs
teams_ids_df = DB.get_all_teams()

teams_ids_df.team1_id.unique().shape

df.set_index('teams_id', inplace=True)
teams_ids_df.set_index('teams_id', inplace=True)

df['teams_name'] = teams_ids_df.loc[df.index, 'teams_name']

df = df.reset_index().set_index('teams_name')
df.rename(columns={'total_goals': 'goals'}, inplace=True)

# preprocess by position (all 10 positions)
teams_df, train_ds, test_ds = TrainingHarness.preprocess_by_position(df)

# preprocess by team (all 20 teams)
teams_df, train_ds, test_ds = TrainingHarness.preprocess_by_team(df)

# train the models
experiment = EurekaRegression.fit(x_y_data=df, training_pred_score_path=training_pred_score_path, type_of_time_series='teams', cv_flag=False, api_key=API_KEY, rest_api_key=REST_API_KEY, param_search='bayes', workspace=workspace)

# predict on the test set
tss, preds, load_path = EurekaRegression.predict(x_y_data=test_ds, training_pred_score_path=training_pred_score_path)

# evaluate the trained models
for model_name in tss.keys():
    
    save_path = os.path.join(training_pred_score_path, f'saved_models/{model_name}_gluonts_metrics.csv')
    
    metrics_df = Evaluation.gluonts_evaluation(tss[model_name], preds[model_name], load_path[model_name], test_ds)
    
    metrics_df.to_csv(save_path)
    
#     experiment.log_asset(save_path)
        
    print(f'{model_name} : \n {metrics_df[["MASE", "abs_error", "MSE"]].head()}\n\n' )

# draft helper method for logging graphs
def plot_log_and_save_graphs(tss, preds, profit_df, saved_models_path, model_name, plot_length=300, prediction_intervals = (50.0, 90.0)):
    

    for index in profit_df.index.values[:5]:  
        
        experiment.log_metrics(profit_df.loc[index, ['mae', 'acc', 'profit', 'profit_rate']].to_dict(), prefix=f'{index}_{model_name}')

        legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        tss[index][-plot_length:].plot(ax=ax)  # plot the time series
        preds[index].plot(prediction_intervals=prediction_intervals, color='g')
        plt.grid(which="both")
        plt.legend(legend, loc="upper left")

        fig.canvas.draw()

        figure_name = f'{index}_{model_name}'

        experiment.log_figure(figure_name=figure_name, figure=fig)

        plt.savefig(os.path.join(saved_models_path, figure_name))


# calculate profit
for model_name in tss.keys():
    
    save_path = os.path.join(training_pred_score_path, f'saved_models/{model_name}_profits.csv')
    
    profit_df = Evaluation.profit_evaluation(test_ds, preds[model_name], teams_df)
    
    profit_df.to_csv(save_path)
        
    print(f'{model_name} : \n {profit_df.head()}\n\n' )
    
    plot_log_and_save_graphs(tss[model_name], preds[model_name], profit_df, saved_models_path, model_name, plot_length=60, prediction_intervals=(50.0, 90.0))

for model_name in tss.keys():
    
    print(f'{model_name} : \n {Evaluation.plot_preds_graph(tss[model_name], preds[model_name], plot_length=60, prediction_intervals=(50.0, 90.0))}\n\n' )


# zip and upload/log the saved_models directory to comet-ml
shutil.make_archive(saved_models_path, 'zip', saved_models_path)

experiment.log_asset(saved_models_path + '.zip')

experiment.end()