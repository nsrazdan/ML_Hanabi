

# importing libs
import tensorflow as tf
import numpy as np

# loading data
import load_data
# data = load_data.main("/data/Hanabi-Full_2_6_150.pkl")

# test data while load_data doesn't work
train_x = np.array([0, 0, 0, 0, 0, 0], dtype = int)
train_y = np.array([1, 1, 1, 1, 1, 1], dtype = int)
test_x = train_x
test_y = train_y


# TODO: have actual values (num_inputs should be 658)
num_inputs = 1
num_hidden_nodes = 64
num_outputs = 2
batch_size = 10
num_epochs = 1
learning_rate = 0.001

# create layers
input_layer = tf.keras.layers.Flatten(input_shape=([num_inputs]))
hidden_layer_1 = tf.keras.layers.Dense(units=num_hidden_nodes, activation='relu')
hidden_layer_2 = tf.keras.layers.Dense(units=num_hidden_nodes, activation='relu')
output_layer = tf.keras.layers.Dense(units=num_outputs, activation = 'softmax')

# instantiate model
model = tf.keras.models.Sequential([input_layer, hidden_layer_1, hidden_layer_2, output_layer])

# compile model and set loss and optimizer func
model.compile(loss='sparse_categorical_crossentropy',
               optimizer=tf.keras.optimizers.Adam(learning_rate),
               metrics=['accuracy'])

# train model
model.fit(train_x, train_y, num_epochs)

# evaluate/test model
model.evaluate(test_x, test_y)
