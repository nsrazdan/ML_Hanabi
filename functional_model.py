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

#CONST VARIABLES

class Naive_MLP(object):
    def __init__(self, train_obs, train_act, valid_obs, valid_act):
        '''
        - data: the object returned from load data
        '''
        self.train_obs = train_obs
        self.train_act = train_act
        self.valid_obs = valid_obs
        self.valid_act = valid_act
        
        
    '''Create a model'''
    def create_model(self, activations=['relu','softmax'], num_hidden_nodes = [256,128,64,20]):
        
        input_layer = Input(shape=(len(self.train_obs),))
        hidden_layer = Dense(num_hidden_nodes[0], activations=activations[0])(input_layer)
        for i in range(1,len(num_hidden_nodes)-1):
            hidden_layer = Dense(num_hidden_nodes[i], activations=activations[0])(hidden_layer)

        output_layer = Dense(num_hidden_nodes[len(num_hidden_nodes)-1], activations = activations[len(activations)-1])
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    '''train the model we just created'''
    def train_model(self, optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 
            loss ='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            epochs=10,
            batch_size=10):
        
        #obtain input and output layer to create the model
        model = self.create_model
        model.compile(optimizer=optimizer,
                loss=loss,
                metrics=metrics)
        tr_history = model.fit(self.train_obs, self.train_act,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=(self.valid_obs,self.valid_act),
                shuffle = True)
        return model
