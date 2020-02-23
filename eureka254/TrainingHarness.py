import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.predictor import Predictor
from gluonts.dataset.util import to_pandas
from gluonts.trainer import Trainer

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import *

from eureka254.ModelKwargs import *
from eureka254.logging_setup import add_logger

from comet_ml import Optimizer
from comet_ml import Experiment
from comet_ml import API
import comet_ml

GluonTSEstimatorKwargs = GluonTSEstimatorKwargs()
GluonTSBayesEstimatorKwargs = GluonTSBayesEstimatorKwargs()

@add_logger
class TrainingHarness:


  """ This class contains methods that make the application of models, model training, and
  model hyperparameter tuning easier.

  Methods ```cv_harness_trainer```, ```score_bayes_trainer_harness``` and  ```score_trainer_harness``` are 
  typically called  in practice, for instance, in the EurekaRegression.py `fit` method.
  See ``fit`` method within :class:`eureka254.EurekaRegression.EurekaRegression.fit`.

  """


  def __init__(self, **kwargs:dict):   


    """
    Parameters
    ----------
    :param: ``kwargs`` : ``dict``
      Arbitrary keyword arguments.

    Args:
    -----
    Keyword arguments. If you do accept ``**kwargs``, make sure
    you link to documentation that describes what keywords are accepted,
    or list the keyword arguments here:

    :param: ``models`` : ``dict``
      A dict with keys being the type of time series and values being a list of
      regression models to train.

    :param: ``x_y_data`` : ``Gluonts ListDataset``
      References the whole time series before preprocessing and training routing/scoring.

    :param: ``type_time_series`` : ``dict``
      Type of time series to train model. eg teams or by position

    :param: ``prediction_length`` : ``str``
      Length of the prediction horizon

    :param: ``start_date`` : ``np.array``
      Start date of the time series

    :param: ``freq`` : ``str eg. '1D', '2H', '3S'...``
      Frequency of the data to train on and predict 

    :param: ``team_initials`` : ``str``
      The team initials to index by if modelling for only one team 

    :param: ``n_splits`` : ``int``
      The number of splits to divide the data into during cross validation

    :param: ``out_path`` : ``str``
      This is the filepath of generated models and scores

    :param: ``experiment`` : ``comet_ml.Experiment``
      The comet_ml object that logs to the cloud

    :param: ``api_key`` : ``str``
      Your comet ml api-key - for logging and bayesian optimization (https://www.comet.ml/signup)
    
    :param: ``rest_api_key`` : ``str``
      Your comet ml REST api-key - for retrieving best hyperparameters (https://www.comet.ml/docs/rest-api/getting-started/)

    :param: ``param_search`` : ``str``
      The type of hyperparameter search to perform

    :param: ``workspace`` : ``str``
      The comet ml workspace where this run's metadata will be stored
      

  """
    self.__dict__.update(kwargs)

    allowed_keys = ["models",
              "train_data",
              "val_data"
              "model_kwarg",
              "type_time_series",
              "prediction_length",
              "start_date",
              "freq",
              "team_initials",
              "n_splits",
              "out_path",
              "experiment",
              "api_key",
              "rest_api_key",
              "param_search",
              "workspace",
              ]


    self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    if not hasattr(self, "models"):
      self.model = GluonTSEstimatorKwargs.AppliedEurekaRegressorModels

    if not hasattr(self, "model_kwarg"):
      self.model_kwarg = GluonTSEstimatorKwargs.DARE

    if not hasattr(self, "train_data"):
      self.train_data = ListDataset 

    if not hasattr(self, "val_data"):
      self.val_data = ListDataset 

    if not hasattr(self, "type_time_series"):
      self.type_time_series = 'teams'

    if not hasattr(self, "prediction_length"):
      self.prediction_length = 50

    if not hasattr(self, "start_date"):
      self.start_date = "2019-04-18"

    if not hasattr(self, "freq"):
      self.freq = "1H"
    if not hasattr(self, "team_initials"):
      self.team_initials = "MNC"

    if not hasattr(self, "n_splits"):
      self.n_splits = 5

    if not hasattr(self, "out_path"):
      self.out_path = os.path.dirname(__file__)

    if not hasattr(self, "api_key"):
      self.api_key = None

    if not hasattr(self, "rest_api_key"):
      self.rest_api_key = None

    if not hasattr(self, "param_search"):
      self.param_search = None

    if not hasattr(self, "trainer"):
      self.trainer = TrainerKwargs.trainer['Trainer']

    if not hasattr(self, "workspace"):
      self.workspace = None

    self.best_params = {}


  @staticmethod
  def _create_dir_if_not_exists(path):
    import os
    if not os.path.exists(path):
      os.makedirs(path)


  def Preprocess_stock_data(self ,data ):
    
    custom_dataset = data.close.values
    custom_dataset = custom_dataset.reshape(1, -1)
    prediction_length = self.prediction_length
    start_date = self.start_date
    freq = self.freq
    start = pd.Timestamp(start_date , freq) 

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start} for x in custom_dataset[:, :-prediction_length]], freq = '1H')
    
    # test datListDatasetaset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start} for x in custom_dataset], freq='1H')
    
    return train_ds, test_ds


  def preprocess_by_single_team(self ,data ):
    self.__log.info(
            "Starting preprocessing time series by a single team before training routine starts" )
    
    team_initials = self.team_initials
    custom_dataset = data[data.index.str.contains(team_initials)]
    custom_dataset = data.goals.values
    custom_dataset = custom_dataset.reshape(1, -1)
    prediction_length = self.prediction_length
    start_date = self.start_date
    freq = self.freq
    
    start = pd.Timestamp(start_date , freq) 

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start} for x in custom_dataset[:, :-prediction_length]], freq = '1H')
    

    # test datListDatasetaset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start} for x in custom_dataset], freq='1H')
    self.__log.info(
            "Finished preprocessing time series by single team" )
    return train_ds, test_ds
    
  def preprocess_by_teams(self, df):
    self.__log.info(
            "Starting preprocessing time series by teams before training routine starts" )

    teams_df = pd.DataFrame()
  
    prediction_length = self.prediction_length
    start_date = self.start_date
    freq = self.freq
    start = pd.Timestamp(start_date , freq)
    pos_df = pd.DataFrame()
    teams = sorted(list(set([x.split('-')[0] for x in df.index.values])))
    for team in teams:
      
      temp_df = df.loc[df.index.str.contains(team)].copy(deep=True)
      teams_df = pd.concat([teams_df,  pd.Series(temp_df.index).to_frame().T])
      temp_df['league_no/week'] = temp_df['league_no'].map(str) + '-' +  temp_df['league_week'].map(str)
      temp_df = temp_df.reset_index().set_index('league_no/week')
      temp_df = temp_df['goals'].to_frame().T

      pos_df = pd.concat([pos_df, temp_df])

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start} for x in pos_df.values[:, :-prediction_length]], freq = '1H')

    # test datListDatasetaset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start} for x in pos_df.values], freq='1H')
    self.__log.info(
            "Finished preprocessing time series by teams" )
    return teams_df, train_ds, test_ds


  def preprocess_by_position(self, df):
    self.__log.info(
            "Starting preprocessing time series by position before training routine starts" )

    teams_df = pd.DataFrame()
  
    prediction_length = self.prediction_length
    start_date = self.start_date
    freq = self.freq
    
    start = pd.Timestamp(start_date , freq)
    
    pos_df = pd.DataFrame()
    for pos in df.week_position.unique():

      temp_df = df.loc[df.week_position==pos].copy(deep=True)
      teams_df = pd.concat([teams_df,  pd.Series(temp_df.index).to_frame().T])
      temp_df['league_no/week'] = temp_df['league_no'].map(str) + '-' +  temp_df['league_week'].map(str)
      temp_df = temp_df.reset_index().set_index('league_no/week')
      temp_df = temp_df['goals'].to_frame().T

      pos_df = pd.concat([pos_df, temp_df])

    # train dataset: cut the last window of length "prediction_length", add "target" and "start" fields
    train_ds = ListDataset([{'target': x, 'start': start} for x in pos_df.values[:, :-prediction_length]], freq = '1H')

    # test datListDatasetaset: use the whole dataset, add "target" and "start" fields
    test_ds = ListDataset([{'target': x, 'start': start} for x in pos_df.values], freq='1H')
    self.__log.info(
            "Finished preprocessing time series by position" )
    return teams_df, train_ds, test_ds

  def _cv_train_model_other(
      self,
      model: object,
      X_train: np.array,
      X_test: np.array,
      model_name: str,
      split_num: int,
      out_path: str,
      type_time_series: str,
  ):

        """ Cross validation training primitive for mxnet/gluon style regression models.
        Parameters
        ----------
        :param: models : `regression model class object`
            Within the training loop, an instantiated model is passed to this method.

        :param: X_train : `np.array`
          Time series array of data used for model training

        :param: X_test : `np.array`
          Time series array of data used for evaluating the model after training

        :param: model_name : `str`
            Machine learning model type.

        :param: split_num : `int`
            The split number within the time series cross validation iterator during cross validation.

        :param: out_path : `str`
            Filepath of generated models and scores.

        :param: type of time series : `str`
              Type of time series to train model. e.g., 'teams'.

        Returns
        -------
        :return: ``y_pred`` : ``np.array``
            An array of values predicted for the test set from the supplied and trained
            regression regression model.

        :return: ``mse`` : ``float``
            The gluonts calculated mean squared error score between the predicted test set `y_pred`
            and the provided ground truth, `y_test`.

        :return: ``rmse_score_`` :``float``
            The gluonts calculated root-mean-squared error value between the predicted
            test set `y_pred` and the provided ground truth, `y_test`.


        """

        status_string = "type_time_series={}, mlmodel={}, mlmodelargs={}".format(
            type_time_series, model_name, models
        )
        status_string_short = "type_time_series={}, mlmodel={}".format(
            type_time_series, model_name
        )
        self.__log.info("Inner function for {}".format(status_string))

        self.__log.info(
            "Starting regression training routine for {}".format(status_string_short)
        )
        
        ml = model.train(X_train)

        self.__log.info("Starting prediction routine.")
        
        
        forecast_it, ts_it = make_evaluation_predictions(
                          dataset=X_test,  # test dataset
                          predictor=ml,  # predictor
                          num_eval_samples=1,  # number of sample paths we want for evaluation
                          )

        forecasts = list(forecast_it)
        y_pred = forecasts[0].mean

        tss = list(ts_it)
        
        # persist model
        self.__log.info("Persisting Model for {}".format(status_string_short))

    
        model_save_name = "{}_{}_{}_cv_trained".format(
          model_name,type_time_series, split_num)

        save_path = os.path.join(out_path, "saved_models", model_save_name)
        self._create_dir_if_not_exists(save_path)

        ml.serialize(Path(save_path))

        # print(to_pandas(test_entry).shape)
        self.__log.info("Scoring for {}".format(status_string_short))
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(X_test))

        
        mse = agg_metrics["MSE"]  
        rmse_score_  = agg_metrics["RMSE"]
        self.__log.info("mse={}".format(mse))
        self.__log.info("rmse_score={}".format(rmse_score_))
        

        # output scoring for cv loop
        return y_pred, mse, rmse_score_

  def cv_harness_trainer(self):


    list_y_pred = []
    list_score = []

    models = self.model
    type_time_series = self.type_time_series

    train_data = self.train_data
    val_data = self.val_data
    n_splits = self.n_splits
    out_path = self.out_path
    freq = self.freq
    prediction_length = self.prediction_length
    start_date = self.start_date
    train_entry = next(iter(train_data))
    val_entry = next(iter(val_data))

    for models_ in models.get(type_time_series):

      for key, value in models_.items():
          model_name = key
          model = value[0]
          model_kwarg = value[1]

          self.__log.info(
                "Starting generic cv train loop for Type_of_time_series={}, mlmodel={}, modelkwarg={}".format(
                    type_time_series, model_name, model_kwarg
                )
            )

          self.__log.info("Regression")
          splits = TimeSeriesSplit(n_splits=n_splits)

          X_train = to_pandas(train_entry).values
          X_val = to_pandas(val_entry).values

          # start the cross validation loop
          self._internal_cv_trainer(
                models=model,
                model_name=model_name,
                X_train=X_train,
                X_test=X_test,
                ss=splits,
                n_splits=n_splits,
                list_y_pred=list_y_pred,
                list_score=list_score,
                out_path=out_path,
                start_date=start_date,
                prediction_length=prediction_length,
                freq=freq,
                type_time_series =type_time_series,

            )

          # output the dataframe of predicted vals with index as sample numbers
          y_pred_list_df = pd.concat(list_y_pred, axis=1)

          # one liner to remove duplicate columns
          y_pred_list_df = y_pred_list_df.loc[
                            :, ~y_pred_list_df.columns.duplicated()
                            ]
          y_pred_list_df.set_index("index", inplace=True)

          
          y_pred_list_df_path_csv = os.path.join(
              out_path,
              "saved_models",
                "y_pred_list_df.csv",
          )

          y_pred_list_df.to_csv(y_pred_list_df_path_csv)

          # output the dataframe of scores with index as sample numbers
          score_list_df = pd.concat(list_score, axis=1)

          # one liner to remove duplicate columns
          score_list_df = score_list_df.loc[:, ~score_list_df.columns.duplicated()]

          score_list_df_path_csv = os.path.join(
              out_path,
              "saved_models", str(type_time_series + "_" + model_name + "_" +  "regression" + "_score_list_df.csv")
              )
          # print(score_list_df)
          # print(score_list_df_path_csv)
          score_list_df.to_csv(score_list_df_path_csv)

    return y_pred_list_df, score_list_df
      
  def _internal_cv_trainer(
      self,
      model_name: str,
      models: object,
      type_time_series: str,
      X: np.array,
      ss: object,
      n_splits: int,
      list_y_pred: list,
      list_score: list,
      out_path: str,
      freq: int,
      prediction_length : int, 
      start_date : str,
      
      
  ):
        """ Cross validation training loop for an individual regression models.

        Parameters
        ----------
        :param: ``X`` : ``np.array``
            Array of time series data.

        :param: ``model_name`` : ``str``
            Machine learning model type.

        :param: ``models`` : ``regression model class object``
            Within the training loop, an instantiated model is passed to this method.

        :param: ``type_time_series`` : ``dict``
            Type of time series to train model. eg teams or by position

        :param: ``prediction_length`` : ``str``
            Length of the prediction horizon

        :param: ``start`` : ``np.array``
            Start date of the time series

        :param: ``freq`` : ``str eg. '1D', '2H', '3S'...``
            Frequency of the data to train on and predict 

        :param: ``ss`` : ``scikit-learn split iterator object``
            This is an instantiated split iterator object to control time series cross validation
            splitting within the cross validator.

        :param: ``n_splits`` : ``int``
            The number of splits to divide the data into during cross validation.

        :param: ``list_y_pred`` : ``list``
            A tracking list entitity for the predicted values within each model cross-validation loop.

        :param: ``list_score`` : ``list``
            A tracking list entitity for the scored values within each model cross-validation loop.

        :param: ``out_path`` : ``str``
            his is the filepath of generated models and scores.


        Returns
        -------
        :return: ``None``

        """

        split_num = np.int(0)
        y_test_indices = []

        y_pred_df = pd.DataFrame()
        y_pred_ = []
        y_true_ = []

        score_df = pd.DataFrame()
        score_1 = []
        score_2 = []
        

        for train_index, test_index in ss.split(X=X):
        
          split_num += 1
          self.__log.info("%%--%%")
          self.__log.info("Cross fold: %i of %i", split_num, n_splits)
          
          # a workaround made here as the test set after split is not utilised. The training set is split 
          dataset, X_test = X[train_index], X[test_index]
          start = pd.Timestamp(start_date, freq)
          X_train = ListDataset([{'target': x, 'start': start} for x in dataset.reshape(1, -1)[:, :-prediction_length]], freq = '1H')
          X_test = ListDataset([{'target': x, 'start': start} for x in dataset.reshape(1, -1) ], freq='1H')
          y_pred_temp, score_1_temp, score_2_temp = self._cv_train_model_other(
                  models=models,
#                     model_kwarg=model_kwarg,
                  X_train=X_train,
                  X_test=X_test,
                  split_num=split_num,
                  out_path=out_path,
                  model_name =model_name,
                  type_time_series =type_time_series

              )

          self.__log.info("Score_1_temp={}".format(score_1_temp))

          dataset_ = next(iter(X_test))
          dataset_pd = to_pandas(dataset_)
          y_test = dataset_pd[-prediction_length:].index
          y_test_indices.append(y_test.values)
          y_pred_.append(y_pred_temp)
          y_true_.append(dataset_pd.loc[y_test].values)

          # keep track of the scores during loops
          score_1.append([score_1_temp])
          score_2.append([score_2_temp])

        y_pred_df[str(model_name + "_" + "regression")] = np.concatenate(
            y_pred_
        ).ravel()

        y_pred_df["index"] = np.concatenate(y_test_indices).ravel()

        y_pred_df[str("y_true_val")] = np.concatenate(y_true_).ravel()

        score_df[str(model_name + "_" + "regression" + "_" + "mse")] = np.concatenate(
            score_1
        ).ravel()

        score_df[str(model_name + "_" + "regression" + "_" + "rmse")] = np.concatenate(
            score_2
        ).ravel()

        score_df[str(model_name + "_" + "regression" + "_" + "global_mse_ave")] = score_df[
            str(model_name + "_" + "regression" + "_" + "mse")
        ].mean()

        score_df[str(model_name + "_" + "regression" + "_" + "global_mse_med")] = score_df[
            str(model_name + "_" + "regression" + "_" + "rmse")
        ].median()

        score_df[str(model_name + "_" + "regression" + "_" + "global_mse_std")] = score_df[
            str(model_name + "_" + "regression" + "_" + "mse")
        ].std()

        score_df[
            str(model_name + "_" + "regression" + "_" + "global_rmse_ave")
        ] = score_df[str(model_name + "_" + "regression" + "_" + "rmse")].mean()

        score_df[str(model_name + "_" + "regression" + "_" + "global_rmse_med")] = score_df[
            str(model_name + "_" + "regression" + "_" + "rmse")
        ].median()

        score_df[str(model_name + "_" + "regression" + "_" + "global_rmse_std")] = score_df[
            str(model_name + "_" + "regression" + "_" + "rmse")
        ].std()

        
        path_score_list_path_csv = os.path.join(
            out_path,
            "saved_models",
            str( str(type_time_series) +
            "_"
             + str(model_name)
                + "_"
                +
                str(split_num)
                + "_"
                + "regression"
                + "_"
            
                + "_cv_score_list.csv"
            ),
        )

        score_df.to_csv(path_score_list_path_csv)

        list_y_pred.append(y_pred_df)
        list_score.append(score_df)


  # separate function for bayes optimization as comet-ml works best when the optimization runs first for all models before the final training (for now)
  def score_bayes_trainer_harness(self):
      """
      Returns
      -------
      :return: ``None``
          None, but results in saved models suitable for scoring and trained
          on all available data.
      """

      self.__log.info("Starting generic score train loop")
      train_data = self.train_data
      val_data = self.val_data
      models = self.model
      out_path = self.out_path
      type_time_series = self.type_time_series
      param_search = self.param_search
      trainer = self.trainer
      api_key = self.api_key
      rest_api_key = self.rest_api_key
      workspace = self.workspace
      

      for models_ in models.get(type_time_series):

        for key, value in models_.items():

          model_name = key
          model = value[0]
          model_kwarg = value[1]

          if param_search == 'bayes':

            search_space = GluonTSBayesEstimatorKwargs.BayesModelLookup.get(model_name)

            # comet-ml hyperparameter optimization configuration (bayes in this case)
            config = {"algorithm": "bayes",
                  "spec": {
                      "maxCombo": 5,  # no of combinations to try
                      "objective": "minimize",
                      "metric": "loss",
                      "seed": 42,
                      "gridSize": 10,
                      "minSampleSize": 100,
                      "retryLimit": 20,
                      "retryAssignLimit": 0,
                  },
                  "name": "My Bayesian Search",
                  "trials": 1,
                  }

            config['parameters'] = search_space

            # current time
            timestr = time.strftime("%Y%m%d-%H%M%S")
            # comet-ml project name for the optimization
            project_name = f"optimizer-{model_name}-{timestr}"
            # initialize the comet-ml optimizer
            optimizer = Optimizer(config=config, api_key=api_key, project_name=project_name)
            # loop through the parameter combinations that the bayes optimizer suggests
            for experiment in optimizer.get_experiments():

                # explicitly set the model parameters (should be generic for any model)
                if model_name == "SimpleFeedForward":
                
                  hidden1 = experiment.get_parameter("hidden_layer_size")
                  hidden2 = experiment.get_parameter("hidden2_layer_size")
                  model_kwarg['num_hidden_dimensions'] = [hidden1, hidden2]

                  self.__log.info(f"model_kwarg['num_hidden_dimensions'] : {model_kwarg['num_hidden_dimensions']}")


                elif model_name == "DeepAREstimate":
                  
                  model_kwarg['num_layers'] = experiment.get_parameter("num_layers")
                  model_kwarg['num_cells'] = experiment.get_parameter("num_cells")
                  model_kwarg['cell_type'] = experiment.get_parameter("cell_type")
                  model_kwarg['dropout_rate'] = experiment.get_parameter("dropout_rate")
                  

                # set trainer params
                trainer.learning_rate = experiment.get_parameter("learning_rate")
                trainer.batch_size = experiment.get_parameter("batch_size")
                trainer.epochs = 2             
                
                # initialize model from the suggested hyperparameters
                model = model.from_hyperparameters(**model_kwarg)     
                # set the trainer
                model.trainer = trainer 
                
                self.__log.info(f'\n model.trainer.lr : {model.trainer.learning_rate}')
                self.__log.info(f'model.trainer.epochs : {model.trainer.epochs}\n')

                # train the model
                predictor = model.train(train_data)
                # make predictions
                forecast_it, ts_it = make_evaluation_predictions(
                                            dataset=val_data,  # test dataset
                                            predictor=predictor,  # predictor
                                            num_eval_samples=1,  # number of sample paths we want for evaluation
                                            )

                # convert gluonts objects to lists
                forecasts = list(forecast_it)
                tss = list(ts_it)

                # get prediction length
                prediction_length = forecasts[0].mean.shape[0]

                y_test_ = list(val_data)[0]['target']
                y_preds_ = forecasts[0].mean
                y_test_ = y_test_[-prediction_length:]

                mae_ = mean_absolute_error(y_test_, y_preds_)

                # Report the loss to comet
                experiment.log_metric("loss", mae_)    

          experiment.end()    

        # initialize comet REST API to retrieve the best hyperparameters
        comet_api = comet_ml.API(rest_api_key=rest_api_key)
      
        project = comet_api.get(workspace=workspace, project_name=optimizer.experiment_kwargs['project_name'].lower())

        # get the experiment ids
        exp_ids = [x.id for x in project]
        
        scores_df = pd.DataFrame(index=exp_ids, columns=['metric'])
        # loop through the experiments within the comet project
        for exp_id in exp_ids:
            
            exp = comet_api.get(f"{workspace}/{project_name.lower()}/{exp_id}")
            
            scores_df.at[exp_id, 'metric'] = exp.get_metrics()[0]['metricValue']
            
        scores_df.metric = scores_df.metric.map(float)
        # get experiment_id of the best score
        best_exp_id = scores_df.metric.idxmin()
        # get the best experiment
        exp = comet_api.get(f"{workspace}/{project_name.lower()}/{best_exp_id}")
        # get the best hyperparameters
        best_params = {x['name']: x['valueCurrent'] for x in exp.get_parameters_summary() if x['name'] != 'f'}
        # save best params in model_name-keyed dictionary for later use
        self.best_params[model_name] = best_params

    
       
  def score_trainer_harness(self):
      """
      Returns
      -------
      :return: ``None``
          None, but results in saved models suitable for scoring and trained
          on all available data.
      """

      self.__log.info("Starting generic score train loop")

      train_data = self.train_data
      val_data = self.val_data
      models = self.model
      out_path = self.out_path
      type_time_series = self.type_time_series
      param_search = self.param_search
      trainer = self.trainer
      api_key = self.api_key
      best_params = self.best_params

      timestr = time.strftime("%Y%m%d-%H%M%S")
      experiment = Experiment(api_key, project_name=f'run-{timestr}')

      self.__log.info(f"best_params : {best_params}")

      for models_ in models.get(type_time_series):

        for key, value in models_.items():

          model_name = key
          model = value[0]
          model_kwarg = value[1]

          # if bayes optimization was performed set the model hyperparameters to the best ones from the search
          if param_search == 'bayes':

            if model_name == "SimpleFeedForward":
                  
              hidden1 = int(best_params[model_name]["hidden_layer_size"])
              hidden2 = int(best_params[model_name]["hidden2_layer_size"])
              model_kwarg['num_hidden_dimensions'] = [hidden1, hidden2]

              self.__log.info(f"model_kwarg['num_hidden_dimensions'] : {model_kwarg['num_hidden_dimensions']}")


            elif model_name == "DeepAREstimate":
              
              model_kwarg['num_layers'] =int(best_params[model_name]["num_layers"])
              model_kwarg['num_cells'] = int(best_params[model_name]["num_cells"])
              model_kwarg['cell_type'] = best_params[model_name]["cell_type"]
              model_kwarg['dropout_rate'] = float(best_params[model_name]["dropout_rate"])
              

            trainer.learning_rate = float(best_params[model_name]["learning_rate"])
            trainer.batch_size = int(best_params[model_name]["batch_size"])
            trainer.epochs = 2             
            
            model = model.from_hyperparameters(**model_kwarg)     
            
            model.trainer = trainer 

          # what should really be logged here
          experiment.log_parameters(model_kwarg, prefix=model_name)


          self.__log.info(
                        "Starting generic score train loop for Type_of_time_series={}, mlmodel={}, modelkwargs={}".format(
                            type_time_series, model_name, model_kwarg
                        )
                    )

          self.__log.info("Regression")


          # for score training, there's nothing to return. It just trains and persists the models
          self._score_train_gluonts_model(
                      model=model,
                      model_name=model_name,
                      type_time_series=type_time_series,
                      train_data=train_data,
                      out_path=out_path,
                          )

      return experiment   
                      
  def _score_train_gluonts_model(self,
                                model,
                                train_data,
                                type_time_series,
                                out_path,
                                model_name,
                                ):     

      predictor = model.train(train_data)

      model_save_name = "{}_{}_score_trained".format(model_name,type_time_series)
      save_path = os.path.join(out_path, "saved_models", model_save_name )
      self._create_dir_if_not_exists(save_path)
      predictor.serialize(Path(save_path) )
      self.__log.info("Training Completed!!!")



      
