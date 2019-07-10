import os, logging
from subprocess import call
import tensorflow as tf
import numpy as np
import load_data
from utils import parse_args, dir_utils
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

#getting rid of "does not support AVX" warnings and info logs
logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# create MLP
class Naive_MLP(object):
    def __init__(self, train_obs, train_act, valid_obs, valid_act):
        self.train_obs = train_obs
        self.train_act = train_act
        self.valid_obs = valid_obs
        self.valid_act = valid_act
