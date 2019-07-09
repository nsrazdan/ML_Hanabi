import tensorflow as tf
import gin
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.models import Model

@gin.configurable
def build_model(obs_size=None, 
        num_hidden_nodes=None,
        hidden_activation=None,
        out_layer_nodes=None,
        out_activation=None):
    #setting up input layer
    input_layer = Input(shape=(obs_size,))

    #setting up all hidden layers
    hidden_layer = Dense(num_hidden_nodes[0], activation=hidden_activation)(input_layer)
    for i in range(1,len(num_hidden_nodes)):
        hidden_layer = Dense(num_hidden_nodes[i], activation=hidden_activation)(hidden_layer)
    flatten_layer = Flatten()(hidden_layer)
    
    #setting up output layer
    output_layer = Dense(out_layer_nodes, activation = out_activation)(flatten_layer)
    #build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
