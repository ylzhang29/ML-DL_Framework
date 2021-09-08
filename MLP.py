import tensorflow as tf
import keras
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

from keras import backend as K
from keras.layers.core import Dense, Activation
from keras import optimizers
from keras.optimizers import SGD , Adam
from keras.optimizers import Adadelta, RMSprop, Adagrad, Adamax
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping,CSVLogger

input_dim = len(X_train[1])
output_dim = len(y_train[1])
print("input", input_dim, "output dimensions", output_dim)
# use determined hyperparameters for the MLP model
layer1 = 500
dropout1 =.5
layer2 = 500
dropout2 =.5
layer3 = 500
dropout3 =.5
lr = .0001
batch_size = 64
opt = Adagrad(lr=lr)

input= Input(shape = (input_dim,))
deep = Dense(units=layer1, activation='selu')(input)
deep = Dropout(dropout1)(deep)
deep = Dense(units=layer2, activation='selu')(deep)
deep = Dropout(dropout2)(deep)
deep = Dense(units=layer3, activation='selu')(deep)
deep = Dropout(dropout3)(deep)
out = Dense(units=output_dim, activation='sigmoid', name= 'out')(deep)
model = Model(input, out)

model.compile(loss=['binary_crossentropy'], metrics=['accuracy'],optimizer=opt)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=50)

history = model.fit(X_train, y_train,
                        epochs=300,
                        batch_size=batch_size,
                        shuffle=True, verbose=2,
                        validation_data=(X_valid,  y_valid), callbacks=[es])

# plot loss during training
import matplotlib.pyplot as plt
fig = plt.title('Losses')
plt.plot(history.history['val_loss'], label='valid')
plt.plot(history.history['loss'], label='train')
plt.legend()
plt.show()

# for binary classification, calculate AUCs
from sklearn.metrics import roc_auc_score
pred_train = model.predict(X_train)
pred_valid = model.predict(X_valid)
auc_train = roc_auc_score(y_train, pred_train)
auc_valid = roc_auc_score(y_valid, pred_valid)
print("train AUC:",auc_train, 'Valid AUC:', auc_valid )
