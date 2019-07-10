import os, logging
from subprocess import call
import tensorflow as tf
import numpy as np
import load_data
import train
import evaluate
import gin
from utils import parse_args, dir_utils
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

# getting rid of "does not support AVX" warnings and info logs
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CONST VARIABLES
DATAPATH = os.path.dirname(os.path.realpath(__file__))

# configuring data to initialize (see mlp.config.gin)
@gin.configurable
def initialize_data(args):
    data = load_data.main(args)
    train_obs, train_act = data.generator(batch_type='train')
    valid_obs, valid_act  = data.generator(batch_type='validation')
    test_obs, test_act = data.generator(batch_type='test')

    return train_obs, train_act, valid_obs, valid_act, test_obs, test_act, data.test_agent

def main():
    # parse arguments
    args = parse_args.parse()
    args.datapath = DATAPATH + "/data/Hanabi-Full_2_6_150.pkl"
    args = parse_args.resolve_datapath(args)

    # external configuration
    gin.external_configurable(tf.keras.optimizers.Adam, module='tensorflow.keras.optimizers')
    gin.external_configurable(tf.keras.losses.mean_squared_error, module='tensorflow.keras.losses')
    gin.parse_config_file('mlp.config.gin')

    # get data, train model, then test model
    train_obs, train_act, valid_obs, valid_act, test_obs, test_act, test_agent = initialize_data(args)
    model = train.main(train_obs, train_act, valid_obs, valid_act, args)
    evaluate.main(model,test_obs, test_act, test_agent)

if __name__ == "__main__":
    main()
