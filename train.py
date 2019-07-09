from utils import parse_args
import importlib
import load_data
import gin
import build_model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

@gin.configurable
class Trainer(object):
    @gin.configurable
    def __init__(self, args,
                optimizer = None,
                 loss=None,
                 metrics=None,
                 batch_size=None,
                 epochs=None):
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.batch_size = batch_size
        self.epochs = epochs

def set_up_vars():
    activations = ['relu', 'softmax']
    num_hidden_nodes = [256,128,64,20]
    
    return activations, num_hidden_nodes

def main(train_obs, train_act, valid_obs, valid_act, args):
    trainer = Trainer(args)
    print("---------CREATING MODEL--------")
    
    '''
    #get the training data and validation data
    activations, num_hidden_nodes = set_up_vars()
     
    # creating layers for model and linking them
    input_layer = Input(shape=(len(train_obs[0]),))
    hidden_layer = Dense(num_hidden_nodes[0], activation=activations[0])(input_layer)
    for i in range(1,len(num_hidden_nodes)-1):
        hidden_layer = Dense(num_hidden_nodes[i], activation=activations[0])(hidden_layer)
    flatten_layer = Flatten()(hidden_layer)
    output_layer = Dense(num_hidden_nodes[len(num_hidden_nodes)-1], activation = activations[len(activations)-1])(flatten_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    '''
    model = build_model.build_model()

    # compiling model
    model.compile(
        optimizer = trainer.optimizer,
        loss = trainer.loss,
        metrics = trainer.metrics)

    print("----------TRAINING MODEL---------")
    # training model
    tr_history = model.fit(train_obs, train_act,
            epochs = trainer.epochs,
            verbose = 1,
            validation_data=(valid_obs,valid_act))

    return model

if __name__ == "__main__":
    args = parse_args.parse_with_resolved_paths()
    gin.parse_config_file(args.configpath)
    main(train_obs, train_act, valid_obs, valid_act, args)
