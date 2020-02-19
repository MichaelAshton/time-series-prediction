from gluonts.model.deepar import DeepAREstimator
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.model.canonical import CanonicalRNNEstimator
from gluonts.model.gp_forecaster import GaussianProcessEstimator
# from gluonts.model.prophet import ProphetEstimator
from gluonts.model.r_forecast import RForecastPredictor
from gluonts.model.seasonal_naive import SeasonalNaiveEstimator
from gluonts.model.seq2seq import Seq2SeqEstimator
from gluonts.trainer import Trainer
from gluonts.block.encoder import Seq2SeqEncoder
from gluonts.distribution import student_t
import mxnet as mx
 
class TrainerKwargs:
  def __init__(self):

    # if len(list(mx.test_utils.list_gpus())) == 0:
    #     self.ctx = 'cpu'        
    # else:
    #     self.ctx = 'gpu'

    self.TR = {
        # "ctx": self.ctx,
        "ctx": 'cpu',
        "epochs": 2,
        "learning_rate":1E-3,
        "hybridize": True,
        "num_batches_per_epoch":10,
        "batch_size": 32,
        "clip_gradient": 10,
        "init": "xavier",
        "learning_rate_decay_factor": 0.5,
        "minimum_learning_rate": 5e-5,
        "num_batches_per_epoch": 50,
        "patience": 10,
        "weight_decay": 1e-8,        
    }
    self.trainer = {
        
       "Trainer" : Trainer(**self.TR)
    }


TrainerKwargs =TrainerKwargs()
tnr = TrainerKwargs.trainer['Trainer']
encode = Seq2SeqEncoder()

class GluonTSBayesEstimatorKwargs:
    def __init__(self):
      
        self.SFF = {
            "learning_rate": {"type": "float", "min": 0.0001, "max": 0.01},#, "scalingType": "loguniform"},
            "hidden_layer_size": {"type": "discrete", "values": [16, 32, 64]},
            "hidden2_layer_size": {"type": "discrete", "values": [4, 8, 16, 32]},
            "batch_size": {"type": "discrete", "values": [16, 32, 64]},
            # "clip_gradient": {"type": "discrete", "values": [10, 20]},
            # "init": {"type": "discrete", "values": ["xavier", "xavier"]},
            # "learning_rate_decay_factor": {"type": "float", "min": 0, "max": 1.0},
            # "minimum_learning_rate" {"type": "float", "min": 0.0001, "max": 0.01},
            # "num_batches_per_epoch": {"type": "discrete", "values": [20, 50]},
            # "patience": {"type": "discrete", "values": [10, 20]},
            # "weight_decay": {"type": "float", "min": 1e-10, "max": 1e-6},
            # "context_length": {"type": "discrete", "values": [50, 100, 150, 200]},
            # "batch_normalization": {"type": "discrete", "values": [True, False]},
            # "mean_scaling": {"type": "discrete", "values": [True, False]},
            # "distr_output": {"type": "discrete", "values": [gluonts.distribution.student_t.StudentTOutput(), gluonts.distribution.student_t.StudentTOutput()]},
        }

        self.DARE = {
            "num_layers": {"type": "discrete", "values": [2, 3, 4]},
            "num_cells": {"type": "discrete", "values": [16, 32, 64]},
            "cell_type": {"type": "categorical", "values": ["lstm", "gru"]},
            "batch_size": {"type": "discrete", "values": [16, 32, 64]},
            "dropout_rate": {"type": "float", "min": 0, "max": 0.5, "scalingType": "uniform"},
            "learning_rate": {"type": "float", "min": 0.0001, "max": 0.01},#, "scalingType": "loguniform"},
            # "clip_gradient": {"type": "discrete", "values": [10, 20]},
            # "init": {"type": "discrete", "values": ["xavier", "xavier"]},
            # "learning_rate_decay_factor": {"type": "float", "min": 0, "max": 1.0},
            # "minimum_learning_rate" {"type": "float", "min": 0.0001, "max": 0.01},
            # "num_batches_per_epoch": {"type": "discrete", "values": [20, 50]},
            # "patience": {"type": "discrete", "values": [10, 20]},
            # "weight_decay": {"type": "float", "min": 1e-10, "max": 1e-6},
            # "context_length": {"type": "discrete", "values": [50, 100, 150, 200]},
            # "embedding_dimension": {"type": "discrete", "values": [10, 20]},
            # "distr_output": {"type": "discrete", "values": [gluonts.distribution.student_t.StudentTOutput(), gluonts.distribution.student_t.StudentTOutput()]},
            # "scaling": {"type": "discrete", "values": [True, False]},
            
            
        }

        self.BayesModelLookup = {
                'SimpleFeedForward': self.SFF,
                'DeepAREstimate': self.DARE,
            }

class GluonTSEstimatorKwargs:
  def __init__(self):
    
       

    self.DARE = {
        "num_layers": 2,
        "num_cells" : 40,
        "cell_type" : "lstm",
        "dropout_rate" : 0.1,
        "context_length": None,
        "scaling": True,
        "distr_output": student_t.StudentTOutput(),
        "embedding_dimension": 20,
        "time_features": None,
        "lags_seq": None,
        "use_feat_dynamic_real": False,
        "use_feat_static_cat": False,
        "cardinality": None,
        "prediction_length": 50,
        "freq": "1H",
        "trainer":tnr,
    }
    self.ModelDARE = {
        "DeepAREstimate" : [DeepAREstimator(**self.DARE), self.DARE],
      }

    self.SFF = {
        "num_hidden_dimensions": [40, 40],        
        "context_length": None,
        "batch_normalization": False,
        "mean_scaling": True,
        "distr_output": student_t.StudentTOutput(),
        "prediction_length": 50,
        "context_length": None,
        "freq": "1H",
        "trainer":tnr,
    }
    self.ModelSFF = {
        "SimpleFeedForward": [SimpleFeedForwardEstimator(**self.SFF), self.SFF],
    }

    
    self.WN = {        
        "prediction_length": 100,
        "freq": "1H",
        "trainer" : tnr
    }
    self.ModelWN = {
        "WaveNet": [WaveNetEstimator(**self.WN), self.WN],           
    }      
        
    self.CRNN = {
        "prediction_length": 380,
        "context_length" : 200,
        "freq": "1H",
        "trainer" : tnr
    }
    self.ModelCRNN = {
        "canonical": [CanonicalRNNEstimator(**self.CRNN), self.CRNN],           
    }

    self.GP = {
        "prediction_length": 380,
        "context_length": None,
        "freq": "1H",
        "cardinality" : 1,
        "trainer" : tnr,
    }
    self.ModelGP = {
        "gaussian": [GaussianProcessEstimator(**self.GP), self.GP],           
    }
    
    # self.PE = {
    #     "prediction_length": 380,
    #     "freq": "1H",
        

    # }
    # self.ModelPE= {
    #     "prophet": ProphetEstimator(**self.PE)
           
    # }
       
          
    # self.RFE = {
    #     "prediction_length": 100,
    #     "freq": "1H",

    # }
    # self.ModelRFE= {
    #     "R" : RForecastPredictor(**self.RFE),
           
    # }

        
    self.SNE = {
        "prediction_length": 380,
        "freq": "1H",
    }
    self.ModelSNE = {
        "seasonalnaive": [SeasonalNaiveEstimator(**self.SNE), self.SNE]           
    }

              
    self.S2Q = {
        "prediction_length": 380,
        "context_length": None,
        "freq": "1H",
        "embedding_dimension" :1,
        "cardinality" :[1],
        "encoder":encode,
        "decoder_mlp_layer":[3],
        "decoder_mlp_static_dim" : 1 ,
        "trainer": tnr,   
    }
    self.ModelS2Q = {
        "seq2seq": [Seq2SeqEstimator(**self.S2Q), self.S2Q]           
    }

    self.AppliedEurekaRegressorModels = {

        "teams" :  [self.ModelSFF, self.ModelDARE], 
        "position" : [self.ModelSFF, self.ModelDARE]
    }