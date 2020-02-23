import os
import json
import numpy as np
import pandas as pd
from pathlib import Path        
from gluonts.evaluation import Evaluator
from eureka254.TrainingHarness import TrainingHarness
from eureka254.ModelKwargs import *
from gluonts.model.predictor import Predictor
from gluonts.dataset.common import ListDataset  
from gluonts.evaluation.backtest import make_evaluation_predictions
from eureka254.logging_setup import add_logger


GluonTSEstimatorKwargs =GluonTSEstimatorKwargs  ()

@add_logger
class EurekaRegression:

    def __init__(self, **kwargs:dict):      

        self.__dict__.update(kwargs)

        allowed_keys = ["models",
                        "x_y_data",
                        "out_path",
                        "model_kwarg",
                        "type_of_timeseries",
                        "prediction_length",
                        ]


        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

        if not hasattr(self, "model"):
            self.models = GluonTSEstimatorKwargs.AppliedEurekaRegressorModels

        if not hasattr(self, "model_kwarg"):
            self.model_kwarg = GluonTSEstimatorKwargs.DARE
            
        if not hasattr(self, "x_y_data"):
            self.x_y_data = ListDataset 
            
        if not hasattr(self, "out_path"):
            self.out_path = os.path.dirname(__file__)

        if not hasattr(self, "team_initials"):
            self.type_of_time_series = 'teams'
                

    @staticmethod
    def _create_dir_if_not_exists(path):
        import pathlib
        pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)

    def fit(self,
        x_y_data: ListDataset,
        training_pred_score_path: str,
        type_of_time_series: str ,
        cv_flag: bool,
        api_key: str,
        rest_api_key: str,
        param_search: str,
        workspace: str,
        ):
        if type_of_time_series == "teams":
            initializer = TrainingHarness()
            teams_df,train_ds, test_ds = initializer.preprocess_by_teams(x_y_data)
            
        elif type_of_time_series == "position":
            initializer = TrainingHarness()
            teams_df, train_ds, test_ds = initializer.preprocess_by_position(x_y_data)
        
        genmod = TrainingHarness(
                train_data=train_ds,
                val_data=test_ds,
                out_path=training_pred_score_path,
                type_of_time_series=type_of_time_series,
                api_key=api_key,
                rest_api_key=rest_api_key,
                param_search=param_search,
                workspace=workspace,
            )
        if cv_flag == True:

            self.__log.info(
            "Entering cross-validation training for Type_of_time_series={}".format(type_of_time_series)
            )
            genmod.cv_harness_trainer()
            genmod.score_trainer_harness()

        else:

            if param_search:
                # separate function for bayes optimization as comet-ml works best when the optimization runs first for all models before the final training (for now)
                genmod.score_bayes_trainer_harness()

            experiment = genmod.score_trainer_harness()

        return experiment


    def predict(
            self,
            x_y_data: ListDataset,
            training_pred_score_path: str
            , 
        ):
            models = self.models
            tss_dict = {}
            forecasts_dict = {}
            load_path_dict = {}

            type_of_time_series = self.type_of_time_series

            for models_ in models.get(type_of_time_series):

                for key, value in models_.items():

                    model_name = key

                    model_save_name = "{}_{}_score_trained".format(model_name, type_of_time_series)
                    load_path = os.path.join(training_pred_score_path, "saved_models" , model_save_name)  

                    ml_loaded =Predictor.deserialize(Path(load_path))

                    forecast_it, ts_it = make_evaluation_predictions(
                                            dataset=x_y_data,  # test dataset
                                            predictor=ml_loaded,  # predictor
                                            num_eval_samples=1,  # number of sample paths we want for evaluation
                                            )

                    forecasts = list(forecast_it)
                    tss = list(ts_it)

                    tss_dict[model_name] = tss
                    forecasts_dict[model_name] = forecasts
                    load_path_dict[model_name] = load_path


            return tss_dict, forecasts_dict , load_path_dict

    
        
    
