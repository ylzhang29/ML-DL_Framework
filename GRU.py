import csv
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import tensorflow as tf
import tensorflow
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, Dense, GRU, LSTM, Reshape, Lambda
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split


def plot_prediction(x, y_true, y_pred):
    """Plots the predictions.

    Arguments
    ---------
    x: Input sequence of shape (input_sequence_length,
    dimension_of_signal)
    y_true: True output sequence of shape (input_sequence_length,
    dimension_of_signal)
    y_pred: Predicted output sequence (input_sequence_length,
    dimension_of_signal)
    """

    plt.figure(figsize=(12, 3))

    output_dim = x.shape[-1]
    for j in range(output_dim):
        past = x[:, j]
        true = y_true[:, j]
        pred = y_pred[:, j]

        label1 = "Seen (past) values" if j == 0 else "_nolegend_"
        label2 = "True future values" if j == 0 else "_nolegend_"
        label3 = "Predictions" if j == 0 else "_nolegend_"

        plt.plot(range(len(past)), past, "o--b", label=label1)
        plt.plot(range(len(past), len(true) + len(past)), true, "x--b", label=label2)
        plt.plot(range(len(past), len(pred) + len(past)), pred, "o--y", label=label3)

    plt.legend(loc='best')
    plt.title("Predictions v.s. true values")
    plt.show()

# read a csv file with longitudinal data (days_data) and common varialbes (PCA_data) in columns
# column numbers are examples, adapt to your own data
with open('dataxxx.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)

data = data[1:302]
days_data = [data[i][57:(57 + 18)] for i in range(0, 301)]
PCA_data = [data[i][125:145] for i in range(0, 301)]

days_data = np.asarray(days_data, dtype=np.float32)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler = scaler.fit(days_data)
days_data = scaler.transform(days_data)

### use this to scale back to original data
############# inversed_days_data = scaler.inverse_transform(normalized_days_data)

PCA_data = np.asarray(PCA_data, dtype=np.float32)
PCA_data = preprocessing.normalize(PCA_data)

# use first 12 days data to predict the next 6 days
days_data_input = np.concatenate((PCA_data, days_data[:, 0:12]), axis=1).reshape(days_data.shape[0], 32, 1)
days_data_predict = days_data[:, 12:18].reshape(days_data.shape[0], 6, 1)

print(days_data_input.shape)
print(days_data_predict.shape)

days_data_input_train, days_data_input_y, days_data_predict_train, days_data_predict_y = train_test_split(
    days_data_input, days_data_predict, test_size=0.1)
days_data_input_test, days_data_input_validation, days_data_predict_test, days_data_predict_validation = train_test_split(
    days_data_input_y, days_data_predict_y, test_size=0.5)

predict_zeros_train = np.zeros(days_data_predict_train.shape)
predict_zeros_test = np.zeros(days_data_predict_test.shape)
predict_zeros_validation = np.zeros(days_data_predict_validation.shape)

print(days_data_input_train.shape)
print(days_data_input_test.shape)
print(days_data_input_validation.shape)

print(days_data_predict_train.shape)
print(days_data_predict_test.shape)
print(days_data_predict_validation.shape)

print(predict_zeros_train.shape)
print(predict_zeros_test.shape)
print(predict_zeros_validation.shape)

# # Seq2Seq model

keras.backend.clear_session()
layers = [35, 35]  # Number of hidden neuros in each layer of the encoder and decoder
learning_rate = 0.01
decay = 0  # Learning rate decay
optimiser = keras.optimizers.Adam(lr=learning_rate,
                                  decay=decay)  # Other possible optimiser "sgd" (Stochastic Gradient Descent)

num_input_features = 1  # The dimensionality of the input at each time step. In this case a 1D signal.
num_output_features = 1  # The dimensionality of the output at each time step. In this case a 1D signal.
# There is no reason for the input sequence to be of same dimension as the ouput sequence.
# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.

loss = "mse"  # Other loss functions are possible, see Keras documentation.

# Regularisation isn't really needed for this application
lambda_regulariser = 0.000001  # Will not be used if regulariser is None
regulariser = None  # Possible regulariser: keras.regularizers.l2(lambda_regulariser)

batch_size = 20
steps_per_epoch = 30  # batch_size * steps_per_epoch = total number of training examples
epochs = 15

input_sequence_length = 32  # Length of the sequence used by the encoder
target_sequence_length = 6  # Length of the sequence predicted by the decoder
num_steps_to_predict = 32  # Length to use when testing the model

num_signals = 2  # The number of random sine waves the compose the signal. The more sine waves, the harder the problem.

# Define an input sequence.
encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

# Create a list of RNN Cells, these are then concatenated into a single layer
# with the RNN layer.
encoder_cells = []
for hidden_neurons in layers:
    encoder_cells.append(
        keras.layers.GRUCell(hidden_neurons, kernel_regularizer=regulariser, recurrent_regularizer=regulariser,
                             bias_regularizer=regulariser))

encoder = keras.layers.RNN(encoder_cells, return_state=True)

encoder_outputs_and_states = encoder(encoder_inputs)
print(encoder_outputs_and_states[0].shape)

# Discard encoder outputs and only keep the states.
# The outputs are of no interest to us, the encoder's
# job is to create a state describing the input sequence.
encoder_states = encoder_outputs_and_states[1:]

# The decoder input will be set to zero (see random_sine function of the utils module).
# Do not worry about the input size being 1, I will explain that in the next cell.
decoder_inputs = keras.layers.Input(shape=(None, 1))

decoder_cells = []
for hidden_neurons in layers:
    decoder_cells.append(
        keras.layers.GRUCell(hidden_neurons, kernel_regularizer=regulariser, recurrent_regularizer=regulariser,
                             bias_regularizer=regulariser))

decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

# Set the initial state of the decoder to be the ouput state of the encoder.
# This is the fundamental part of the encoder-decoder.
decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

# Only select the output of the decoder (not the states)
decoder_outputs = decoder_outputs_and_states[0]

# Apply a dense layer with linear activation to set output to correct dimension
# and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
decoder_dense = keras.layers.Dense(num_output_features, activation='linear', kernel_regularizer=regulariser,
                                   bias_regularizer=regulariser)

decoder_outputs = decoder_dense(decoder_outputs)

model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
model.compile(optimizer=optimiser, loss=loss)

# plot the model
from keras import utils
import pydot
keras.utils.plot_model(model, to_file='model_seq2seq.png', show_shapes=True)

history = model.fit([days_data_input_train, predict_zeros_train], days_data_predict_train, batch_size=32, epochs=50,
                    validation_data=(
                    [days_data_input_validation, predict_zeros_validation], days_data_predict_validation))


# generate test prediction and check for accuracy (mse) and plot the prediction vs actual observations
y_pred = model.predict([days_data_input_test, predict_zeros_test])
def mse(y_true, y_pred):
    from tensorflow.keras import backend
    return backend.mean(backend.square(y_pred - y_true), axis=-1)
print(np.mean(mse(days_data_predict_test, y_pred), axis=0))

for index in range(0, y_pred.shape[0]):
    plot_prediction(days_data_input_test[index, :, :], days_data_predict_test[index, :, :], y_pred[index, :, :])

# predict addtional days and plot the non-scaled observations
Original_Data_input = days_data_input_test.reshape(days_data_input_test.shape[0], days_data_input_test.shape[1])[:,
                      20:32]
Predict_Data_input = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
Predic_True_Data = days_data_predict_test.reshape(days_data_predict_test.shape[0], days_data_predict_test.shape[1])

Predicted_Series = np.concatenate((Original_Data_input, Predict_Data_input), axis=1)

Predicted_Series_days_data = scaler.inverse_transform(Predicted_Series)

True_Series = np.concatenate((Original_Data_input, Predic_True_Data), axis=1)

True_Series_days_data = scaler.inverse_transform(True_Series)

# def plot_prediction(x, y_true, y_pred):
for i in range(0, 15):
    plt.figure(figsize=(12, 3))

    plt.plot(Predicted_Series_days_data[i], "o--y")
    plt.plot(True_Series_days_data[i], "x--b")

    plt.title("Predictions v.s. true values")
    plt.show()
