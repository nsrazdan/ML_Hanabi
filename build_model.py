import tensorflow as tf
import gin
from keras.optimizers import Adam
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def print_info(obs_size, num_hidden_nodes, hidden_activation, out_layer_nodes, out_activation):
    print("----------CREATING MODEL----------")
    print("num_hidden_nodes = ", num_hidden_nodes)
    print("hidden_actiavtion = ", hidden_activation)
    print("out_layer_nodes = ", out_layer_nodes)
    print("out_actiavtion = ", out_activation)

@gin.configurable
def build_model(obs_size=None, 
        num_hidden_nodes=None,
        hidden_activation=None,
        out_layer_nodes=None,
        out_activation=None,
        dropout_rate=None):

    # printing out building model info
    print_info(obs_size, num_hidden_nodes, hidden_activation, out_layer_nodes, out_activation)
    #setting up input layer
    input_layer = Input(shape=(obs_size,))
    #setting up all hidden layers
    hidden_layer = Dense(num_hidden_nodes[0], activation=hidden_activation)(input_layer)
    hidden_layer = Dropout(rate=dropout_rate)(hidden_layer)
    for i in range(1,len(num_hidden_nodes)):
        hidden_liayer = Dense(num_hidden_nodes[i], activation=hidden_activation)(hidden_layer)
        hidden_layer = Dropout(rate=dropout_rate)(hidden_layer)
    #setting up output layer
    output_layer = Dense(out_layer_nodes, activation = out_activation)(hidden_layer)
    #build the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
