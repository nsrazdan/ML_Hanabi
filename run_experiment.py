import os, logging
from subprocess import call
import tensorflow as tf
import numpy as np
import load_data
import functional_model
import train
import evaluate
import gin
from utils import parse_args, dir_utils
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

#getting rid of "does not support AVX" warnings and info logs
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#CONST VARIABLES
DATAPATH = os.path.dirname(os.path.realpath(__file__))

def initialize_data(args):
    data = load_data.main(args)
    train_obs, train_act = data.generator('train')
    valid_obs, valid_act  = data.generator('validation')
    test_obs, test_act = data.generator('test')

    return train_obs, train_act, valid_obs, valid_act, test_obs, test_act

def main():
    #parse arguments
    args = parse_args.parse()
    args.datapath = DATAPATH + "/data/Hanabi-Full_2_6_150.pkl"
    args = parse_args.resolve_datapath(args)

      gin.external_configurable(tf.keras.optimizers.Adam, module='tensorflow.keras.optimizers')
    gin.external_configurable(tf.keras.losses.mean_squared_error, module='tensorflow.keras.losses')
    gin.parse_config_file('mlp.config.gin')

    #create/load data
    '''
    - data: a reference to the Dataset object (refer to load_data.py)
     '''
    train_obs, train_act, valid_obs, valid_act, test_obs, test_act = initialize_data(args)
    model = train.main(train_obs, train_act, valid_obs, valid_act, args)
    evaluate.main(model,test_obs, test_act)

if __name__ == "__main__":
    main()
