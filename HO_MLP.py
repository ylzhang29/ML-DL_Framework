import hyperopt
import tensorflow as tf
import keras
from keras.layers import concatenate, Lambda
from hyperopt import fmin, hp
import sys
os.environ["KERAS_BACKEND"] = "tensorflow"

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, MultiLabelBinarizer
import pickle
from scipy import stats

from hyperopt import Trials, STATUS_OK, tpe
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

space = {
    'layer1': hp.quniform('layer1', 10, 1000, 10),
    'dropout1' : hp.uniform('dropout1', .1, 1),
    'layer2': hp.quniform('layer2', 10, 1000, 10),
    'dropout2' : hp.uniform('dropout2', .1, 1),
    'layer3': hp.quniform('layer3', 10, 1000, 10),
    'dropout3' : hp.uniform('dropout3', .1, 1),
    'batch_size': hp.quniform('batch_size', 5, 500, 5),
    # 'n_epochs' :  hp.quniform('n_epochs',50,1000 ,5),
    'optimizer': hp.choice('optimizer', [optimizers.Adam, optimizers.SGD, optimizers.RMSprop, optimizers.Adagrad,
                                         optimizers.Adadelta, optimizers.Adamax, optimizers.Nadam]),
    'lr': hp.uniform('lr', 0.000001, 0.01),
    'act': hp.choice('act', ['relu', 'selu']),
}

def f_nn(params):
    print('Params testing: ', params)
    input = Input(shape=(input_dim,))
    fc = Dense(units=int(params['layer1']), activation=params['act'], name ="fc1")(input_img)
    fc = Dropout(params['dropout1'])(fc)
    fc2 = Dense(units=int(params['layer2']), activation=params['act'], name ="fc2")(fc)
    fc2 = Dropout(params['dropout2'])(fc2)
    fc3 = Dense(units=int(params['layer3']), activation=params['act'], name ="fc3")(fc2)
    fc3 = Dropout(params['dropout3'])(fc3)
    out = Dense(units=1, activation='sigmoid', name = 'out')(fc3)
    model = Model(input, out)
    opt = params['optimizer'](lr=params['lr'])

    model.compile(loss=['binary_crossentropy'], metrics=['accuracy'],optimizer=opt)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)

    History = model.fit(X_train, y_train,
                        epochs=400,  # int(params['n_epochs']),
                        batch_size=int(params['batch_size']),
                        shuffle=True, verbose=2,
                        validation_data=(X_valid, y_valid), callbacks=[es])

    return {'loss': History.history['val_loss'][-1],
            'train_loss': History.history['loss'][-1],
            'status': STATUS_OK, 'params': params}

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)

with open("trial_obj.pkl", "wb") as f:
    pickle.dump(trials, f, -1)

# save the search results
f = open("HO_results.log", "w")
for i, tr in enumerate(trials.trials):
    f.write("Trial; " + str(i) + ";" + "train_loss;" + str(tr['result']['train_loss'])
            + ";" + "valid_loss; " + str(tr['result']['loss']) + ";"
            +";" + "Parameters; " + str(tr['result']['params']) + "\n")
f.close()

print("*" * 150)
print(best)
print(hyperopt.space_eval(space, best))
print("*" * 150)

