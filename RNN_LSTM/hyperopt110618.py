# This piece of code can be used to limit the use of the GPUs if necessary
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
keras.backend.set_session(session)

import hyperopt
import pickle
import configparser
from model import SunyRnnPreproc, SunyRnnModel


def objective(args):
    params = {'nr_lstm_units': args['nr_lstm_units'],
              'nr_emb_units': args['nr_emb_units'],
              'batch_size': args['batch_size'],
              'trial_nr': trial_nr
              }

    valid_loss = study.run_hyperopt(params)
    return valid_loss


def optimize():
    global trial_nr

    save_trial = 1
    max_trials = 1

    space = {
        'nr_lstm_units': hyperopt.hp.choice('nr_lstm_units', [20, 40, 50, 60]),
        'nr_emb_units': hyperopt.hp.choice('nr_emb_units', [[5, 1,5,1,1], [10, 1,10,1,1],[20, 1,20,1,1], [5, 2,5,2,2], [10, 2,10,2,2], [20, 2, 20, 2, 2], [5,3,5,3,3], [10, 3,10,3,3], [20, 3, 20, 3, 3]]),
        'batch_size': hyperopt.hp.choice('batch_size', [16])
    }

    try:
        trials = pickle.load(open("trial_obj.pkl", "rb"))
        max_trials = len(trials.trials) + save_trial
        trial_nr = max_trials
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, save_trial))
    except:
        trials = hyperopt.Trials()

    best_model = hyperopt.fmin(objective, space, algo=hyperopt.tpe.suggest, trials=trials, max_evals=max_trials)

    with open("trial_obj.pkl", "wb") as f:
        pickle.dump(trials, f)

    print("*" * 150)
    print('best model: {}'.format(hyperopt.space_eval(space, best_model)))
    print("*" * 150)
    f = open("trials.log", "w")
    for i, tr in enumerate(trials.trials):
        trial = tr['misc']['vals']
        for key in trial.keys():
            trial[key] = trial[key][0]
        f.write("Trial no. : %i\n" % i)
        f.write(str(hyperopt.space_eval(space, trial)) + "\n")
        f.write("Loss : " + str(tr['result']['loss']) + ", ")
        f.write("*" * 100 + "\n")
    f.close()


def main():
    global study
    configuration = configparser.ConfigParser()
    configuration.read("config.ini")
    preproc_data = SunyRnnPreproc(configuration)
    study = SunyRnnModel(configuration, preproc_data, hyperopt=True)

    while True:
        optimize()


if __name__ == '__main__':
    study = None
    trial_nr = 1
    main()
