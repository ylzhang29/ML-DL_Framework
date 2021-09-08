# generate stable ensemble MLP predictions
#
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping,CSVLogger
from keras.optimizers import RMSprop
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample
import matplotlib.pyplot as pyplot
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

input_dim = len(features[1])
output_dim = 1

#Hyper-Parameters for the base MLP model
layer1 = 100
layer2 = 300
layer3 = 200
dropout1 = .3
dropout2 = .3
dropout3 = .4
batch_size = 32
opt = RMSprop(lr =.00005, rho=.7)
es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

def evaluate_model(trainX, trainy, x_valid, y_valid):
    # define model
    input_img = Input(shape=(input_dim,))
    deep = Dense(units=layer1, activation='relu')(input_img)
    deep = Dropout(dropout1)(deep)
    deep = Dense(layer2, activation='relu')(deep)
    deep = Dropout(dropout2)(deep)
    deep = Dense(units=layer3, activation='relu')(deep)
    deep = Dropout(dropout3)(deep)
    outlayer = Dense(units=output_dim, activation='sigmoid')(deep)
    model = Model(input_img, outlayer)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['AUC'])
    model.fit(trainX, trainy, epochs=2000,
              verbose=0, batch_size=batch_size,
              validation_data=(x_valid, y_valid),
              callbacks=[es])
    # evaluate the model
    test_loss, test_auc = model.evaluate(x_valid, y_valid, verbose=0)
    print("model and acc", model, test_auc)
    return model, test_auc

# multiple randomly initialized models
X, x_valid, x_test = features, valid_features, test_features
y, y_valid, y_test = target.reshape(-1,1), valid_target.reshape(-1,1), test_target.reshape(-1,1)

n_models = 100
members, AUC_t, AUC_tr,AUC_v, ensemble_AUC_tr, ensemble_AUC_t,ensemble_AUC_v= list(), list(), list(), list(), list(), list(),list()
predict_v, predict_t, predict_tr, predict_all = np.zeros((y_valid.shape[0],0)), np.zeros((y_test.shape[0],0)),
                                                np.zeros((y_train.shape[0],0)),  np.zeros((y_all.shape[0],0))

for _ in range(n_models):
    # select indexes for a subset of training examples
    ix = [i for i in range(len(X))]
    train_ix = resample(ix, replace=True, stratify=y)
    trainX, trainy = X[train_ix], y[train_ix]
    # evaluate model on the subset
    model, valid_auc = evaluate_model(trainX, trainy, x_valid, y_valid)
    AUC_v.append(valid_auc)
    members.append(model)
    # predict model
    yhat_valid = model.predict(x_valid)
    yhat_train = model.predict(x_train)
    yhat_t = model.predict(x_test)
    train_auc = roc_auc_score(y_train, yhat_train)
    test_auc = roc_auc_score(y_test, yhat_t)
    AUC_tr.append(train_auc)
    AUC_t.append(test_auc)
    print("single model AUC: train ", train_auc, ", valid ", valid_auc, ", and test ", test_auc )
    predict_v = np.hstack((predict_v, yhat_valid)) # save predicted valid results
    pred_mean_v = np.mean(predict_v, axis=1) # save final ensemble predictions
    final_auc_v = roc_auc_score(y_valid, pred_mean_v)

    predict_tr = np.hstack((predict_tr, yhat_train)) # save predicted train results
    pred_mean_tr = np.mean(predict_tr, axis=1)
    final_auc_tr = roc_auc_score(y_train, pred_mean_tr)

    predict_t = np.hstack((predict_t, yhat_t)) # save predicted test results
    pred_mean_t = np.mean(predict_t, axis=1)
    final_auc_t = roc_auc_score(y_test, pred_mean_t)
    print("ensemble AUCs: train ", final_auc_tr, ", valid ", final_auc_v, ", and test ", final_auc_t)
    ensemble_AUC_v.append(final_auc_v)
    ensemble_AUC_tr.append(final_auc_tr)
    ensemble_AUC_t.append(final_auc_t)

# plot score vs number of ensemble members
from numpy import mean, std
print("Summary:")
print('Single Model train AUCs-Mean and SD:  %.3f (%.3f)' % (mean(AUC_tr), std(AUC_tr)))
print('Single Model valid AUCs-Mean and SD:  %.3f (%.3f)' % (mean(AUC_v), std(AUC_v)))
print('Single Model test AUCs-Mean and SD:  %.3f (%.3f)' % (mean(AUC_t), std(AUC_t)))

print('Ensemble train AUC:', ensemble_AUC_tr[-1])
print('Ensemble valid AUC:', ensemble_AUC_v[-1])
print('Ensemble test AUC:', ensemble_AUC_t[-1])

pyplot.title("Ensemble Model AUCs")
x_axis = [i for i in range(1, n_models + 1)]
pyplot.plot(x_axis, AUC_tr, marker='o',markersize = 7, color ="r", linestyle='None', label='Single Model Training AUC')
pyplot.plot(x_axis, AUC_v, marker='o',markersize = 7, color ="g", linestyle='None', label='Single Model Validation AUC')
pyplot.plot(x_axis, AUC_t, marker='v',markersize = 7, color ="k", linestyle='None', label='Single Model Test AUC')
pyplot.plot(x_axis, ensemble_AUC_tr, marker= 'o', markersize = 3, color ="r",  label='Ensemble Training')
pyplot.plot(x_axis, ensemble_AUC_v, marker='o', markersize = 3, color ="g",  label='Ensemble Validation')
pyplot.plot(x_axis, ensemble_AUC_t, marker='v', markersize = 3, color ="k",  label='Ensemble Test')
pyplot.legend(loc='best')
pyplot.show()

