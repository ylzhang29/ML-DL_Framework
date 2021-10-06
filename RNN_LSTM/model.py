import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
session = tf.Session(config=config)
keras.backend.set_session(session)

from keras.models import Model
from keras.layers import Input, LSTM, Dense, concatenate, TimeDistributed
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import ast
import configparser
import pickle


class SunyRnnPreproc:

    def __init__(self, config):
        self.data_dir = config['files']['data_dir']
        self.long_file = config['files']['long_file']
        self.pred_file = config['files']['pred_file']
        self.so_feature_files = [f.strip() for f in config['files']['so_feature_files'].split(',')]
        self.so_features_to_normalize = ast.literal_eval(config['features']['so_features_to_normalize'])
        self.so_features_to_one_hot_encode = ast.literal_eval(config['features']['so_features_to_one_hot_encode'])
        self.age_range = [int(a.strip()) for a in config['settings']['age_range'].split(',')]
        self.nr_years = self.age_range[1] - self.age_range[0] + 1
        self.outlook_years = [int(a.strip()) for a in config['settings']['outlook_years'].split(',')]
        self.diagnosis = config['features']['diag_of_interest']
        self.new_test_only = config.getboolean('settings', 'new_test_only')
        self.ages_with_so_features = [int(a.strip()) for a in
                                      config['settings']['ages_with_stand_alone_features'].split(',')]
        self.ages_to_cut_for_y = self.create_y_cut_ages()
        self.age_brackets, self.nr_outputs, self.so_input_needed = self.create_age_brackets()
        if config.getboolean('files', 'data_already_preprocessed'):
            print('loading preprocessed data from file')
            self.x_data_yearly, self.subject_ids, self.x_data_so, self.y_data = \
                pickle.load(open(os.path.join(self.data_dir, 'data_preprocessed.pkl'), 'rb'))
            self.nr_subjects = len(self.subject_ids)
        else:
            print('preprocessing yearly features')
            self.x_data_yearly, self.subject_ids, self.nr_subjects = self.create_x_data_yearly()
            print('preprocessing stand alone features')
            self.x_data_so = self.create_x_data_so()
            print('preprocessing y data')
            self.y_data = self.create_y_data()
            print('saving preprocessed data to file')
            pickle.dump((self.x_data_yearly, self.subject_ids, self.x_data_so, self.y_data),
                        open(os.path.join(self.data_dir, 'data_preprocessed.pkl'), 'wb'))

        self.train_idx, self.val_idx, self.test_idx = self.get_idx()
        print('creating data for the model')
        self.x_train, self.y_train = self.create_data_for_model(self.train_idx)
        self.x_val, self.y_val = self.create_data_for_model(self.val_idx)
        self.x_test, self.y_test = self.create_data_for_model(self.test_idx)

        self.pos_weight = sum([data.size for data in self.y_train]) / sum([data.sum() for data in self.y_train])

    def create_data_for_model(self, ids_in):
        nr_subs = len(ids_in)
        age_diffs = [b[1] - b[0] + 1 for b in self.age_brackets]
        nr_features_per_year = len(self.x_data_yearly.columns) - 2

        x_data_yearly = self.x_data_yearly[self.x_data_yearly.psudoid.isin(ids_in)]
        x_data_yearly_by_age = [x_data_yearly.query('age >= {} & age <= {}'.format(age[0], age[1]))
                                 for age in self.age_brackets]
        x_data_yearly_values = [x_data_yearly_by_age[i].iloc[:,2:].values.reshape(nr_subs, age_diffs[i],
                                 nr_features_per_year) for i in range(len(x_data_yearly_by_age))]

        x_data_so = [self.x_data_so[s][self.x_data_so[s].psudoid.isin(ids_in)].iloc[:, 1:].values
                      for s in self.ages_with_so_features]

        y_data_raw = self.y_data[self.y_data.psudoid.isin(ids_in)]
        y_data = []
        for b in range(len(self.age_brackets)):
            y_train_age = y_data_raw.query('age >= {} & age <= {}'.format(self.age_brackets[b][0], self.age_brackets[b][1]))
            y_train_age_values = y_train_age.iloc[:, -self.nr_outputs[0]:].values
            y_data.append(y_train_age_values[:, :self.nr_outputs[b]].reshape(nr_subs, age_diffs[b], self.nr_outputs[b]))

        return x_data_yearly_values + x_data_so, y_data

    def create_x_data_yearly(self):

        yearly_features = pd.read_csv(os.path.join(self.data_dir, self.long_file))
        subject_ids = yearly_features.psudoid.unique()
        nr_subjects = len(subject_ids)

        all_data = pd.DataFrame({'help_idx': np.arange(self.nr_years * nr_subjects)})
        all_data['psudoid'] = all_data.help_idx.apply(lambda x: subject_ids[int(x / self.nr_years)])
        all_data['age'] = all_data.help_idx.apply(lambda x: x % self.nr_years + self.age_range[0])

        all_data = pd.merge(left=all_data, right=yearly_features, how='left', on=['psudoid', 'age'], copy=False)

        #ndep = all_data[['psudoid', 'age', 'ndep']].copy()
        #ndep = ndep.groupby('psudoid').apply(lambda group: group.interpolate())
        #ndep = ndep.fillna(0)

        all_data = all_data.fillna(0)
        #all_data.ndep = ndep.ndep.copy()
        #all_data = all_data.drop(columns=['help_idx', 'total_category', 'total_incidence'])

        return all_data, subject_ids, nr_subjects

    def create_x_data_so(self):

        all_so_features = {}
        for sof in self.ages_with_so_features:
            sof_idx = self.ages_with_so_features.index(sof)
            so_features = pd.read_csv(os.path.join(self.data_dir, self.so_feature_files[sof_idx]))
            for column in self.so_features_to_normalize[sof_idx]:
                normalizer = StandardScaler()
                normalizer.fit(so_features[column].values.reshape(-1, 1))
                so_features[column] = normalizer.transform(so_features[column].values.reshape(-1, 1))
            for column in self.so_features_to_one_hot_encode[sof_idx]:
                values = so_features[column].unique()
                for value in values:
                    so_features['{}_{}'.format(column, value)] = so_features[column].apply(lambda x:
                                                                                           1 if x == value else 0)
            so_features = so_features.drop(columns=self.so_features_to_one_hot_encode[sof_idx])

            all_so_features[sof] = so_features

        return all_so_features

    def create_y_data(self):

        input_data = pd.read_csv(os.path.join(self.data_dir, self.pred_file))
        all_data = pd.DataFrame({'help_idx': np.arange(self.nr_years * self.nr_subjects)})
        all_data['psudoid'] = all_data.help_idx.apply(lambda x: self.subject_ids[int(x / self.nr_years)])
        all_data['age'] = all_data.help_idx.apply(lambda x: x % self.nr_years + self.age_range[0])

        all_data = pd.merge(left=all_data, right=input_data, how='left', on=['psudoid', 'age'], copy=False)
        all_data = all_data.fillna(0)

        data_out = self.get_y_data_for_sub(all_data, self.subject_ids[0])
        for sub in self.subject_ids[1:]:
            data_sub = self.get_y_data_for_sub(all_data, sub)
            data_out = data_out.append(data_sub)

        return data_out

    def get_y_data_for_sub(self, all_y_data, sub_id):

        data_sub = all_y_data.query('psudoid == {}'.format(sub_id)).copy()
        for oy in self.outlook_years:
            if oy == 1:
                data_sub['t1_{}'.format(self.diagnosis)] = data_sub['t_{}'.format(self.diagnosis)].shift(-1)
            else:
                data_sub['t{}_{}'.format(oy, self.diagnosis)] = np.nan
                for idx in data_sub.index.values[:-oy]:
                    data_sub.at[idx, 't{}_{}'.format(oy, self.diagnosis)] = data_sub.loc[idx + 1:idx + oy,
                                                                            't_{}'.format(self.diagnosis)].max()
        return data_sub

    def get_idx(self):
        shuffled_subids = self.subject_ids.copy()
        np.random.seed(1234)
        np.random.shuffle(shuffled_subids)

        if self.new_test_only:
            nr_train = 0
            nr_val = 0
        else:
            nr_train = int(0.7*self.nr_subjects)
            nr_val = int(0.2*self.nr_subjects)
        train_idx_out = shuffled_subids[:nr_train]
        val_idx_out = shuffled_subids[nr_train:nr_train+nr_val]
        test_idx_out = shuffled_subids[nr_train+nr_val:]

        return train_idx_out, val_idx_out, test_idx_out

    def create_age_brackets(self):
        brackets = []
        nr_outputs = []
        so_inputs = []

        start_age = self.age_range[0]
        current_nr_outputs = len(self.outlook_years)
        add_so = False

        for a in range(self.age_range[0], self.age_range[1]):
            if a + 1 in self.ages_with_so_features or a + 1 in self.ages_to_cut_for_y:
                brackets.append((start_age, a))
                start_age = a+1
                nr_outputs.append(current_nr_outputs)
                if a + 1 in self.ages_to_cut_for_y:
                    current_nr_outputs -= 1
                so_inputs.append(add_so)
                if a + 1 in self.ages_with_so_features:
                    add_so = True
                else:
                    add_so = False

        return brackets, nr_outputs, so_inputs

    def create_y_cut_ages(self):
        ages_out = []
        for oy in self.outlook_years[::-1]:
            ages_out.append(self.age_range[1] - oy + 1)

        return ages_out


class SunyRnnModel:

    def __init__(self, config, dataset, hyperopt=False):
        self.age_range = [int(a.strip()) for a in config['settings']['age_range'].split(',')]
        self.nr_years = self.age_range[1] - self.age_range[0] + 1
        self.outlook_years = [int(a.strip()) for a in config['settings']['outlook_years'].split(',')]
        self.diagnosis = config['features']['diag_of_interest']
        self.base_output_dir = config['files']['output_dir']
        self.ages_with_so_features = [int(a.strip()) for a in
                              config['settings']['ages_with_stand_alone_features'].split(',')]
        self.dataset = dataset
        self.age_brackets = self.dataset.age_brackets
        self.nr_outputs = self.dataset.nr_outputs
        self.so_input_needed = self.dataset.so_input_needed
        self.model_outputs = []
        if not os.path.isdir(self.base_output_dir):
            os.makedirs(self.base_output_dir)

        if hyperopt:
            self.output_dir = None
            self.nr_lstm_units = None
            self.so_embedding_sizes = None
            self.batch_size = None
            self.yearly_inputs, self.so_inputs = None, None
            self.model = None

        else:
            self.output_dir = self.base_output_dir
            self.nr_lstm_units = int(config['settings']['nr_lstm_units'])
            self.so_embedding_sizes = [int(s.strip()) for s in config['settings']['so_embedding_sizes'].split(',')]
            self.batch_size = int(config['settings']['batch_size'])
            self.yearly_inputs, self.so_inputs = self.create_model_inputs()
            self.model = self.create_model()

    def create_model(self):
        output_last_year_prev_rnn, cell_state_prev_rnn, so_idx = self.create_model_for_bracket(0, 0)
        for b_idx in range(1, len(self.age_brackets)):
            output_last_year_prev_rnn, cell_state_prev_rnn, so_idx = self.create_model_for_bracket(b_idx, so_idx,
                                                                                                   output_last_year_prev_rnn,
                                                                                                   cell_state_prev_rnn)

        return Model(inputs=self.yearly_inputs + self.so_inputs, outputs=self.model_outputs)

    def create_model_for_bracket(self, bracket_idx, so_idx, output_last_year_prev_rnn=None, cell_state_prev_rnn=None):
        layer_name = '{}-{}'.format(self.age_brackets[bracket_idx][0], self.age_brackets[bracket_idx][1])
        sub_rnn = self.create_sub_rnn(layer_name)
        predictions = self.create_predictions(layer_name, self.nr_outputs[bracket_idx])

        if bracket_idx == 0:
            output_sub_rnn, output_last_year, cell_state = sub_rnn(self.yearly_inputs[bracket_idx])

        else:
            if self.so_input_needed[bracket_idx]:
                input_to_next_rnn = self.create_so_integration(layer_name, so_idx, output_last_year_prev_rnn)
                so_idx += 1
            else:
                input_to_next_rnn = output_last_year_prev_rnn

            output_sub_rnn, output_last_year, cell_state = sub_rnn(self.yearly_inputs[bracket_idx],
                                                                   initial_state=[input_to_next_rnn,
                                                                                  cell_state_prev_rnn])

        self.model_outputs.append(predictions(output_sub_rnn))

        return output_last_year, cell_state, so_idx

    def create_model_inputs(self):
        nr_features_per_year = len(self.dataset.x_data_yearly.columns) - 2
        so_feature_sizes = [len(self.dataset.x_data_so[s].columns) - 1 for s in self.ages_with_so_features]
        yearly_features_inputs = [
            Input(shape=((b[1] - b[0] + 1), nr_features_per_year), name='input_for_{}-{}'.format(b[0], b[1])) for b in
            self.age_brackets]
        so_features_inputs = [
            Input(shape=(so_feature_sizes[f],), name='so_input_for_{}'.format(self.ages_with_so_features[f])) for f
            in range(len(self.ages_with_so_features))]

        return yearly_features_inputs, so_features_inputs

    def create_sub_rnn(self, layer_name):
        return LSTM(self.nr_lstm_units, return_sequences=True, return_state=True, name='lstm' + layer_name)

    def create_predictions(self, layer_name, nr_output_nodes):
        return TimeDistributed(Dense(nr_output_nodes, activation='sigmoid'), name='predictions' + layer_name)

    def create_so_integration(self, layer_name, so_idx, output_last_year_prev_rnn):
        so_embedding = Dense(self.so_embedding_sizes[so_idx], name='so_embedding' + layer_name)(
            self.so_inputs[so_idx])
        so_concat = concatenate([output_last_year_prev_rnn, so_embedding], name='so_concat' + layer_name)
        input_to_next_rnn = Dense(self.nr_lstm_units, name='so_merge' + layer_name)(so_concat)
        return input_to_next_rnn

    def train(self):
        logger = CSVLogger(os.path.join(self.output_dir, 'logfile.csv'))
        saver = ModelCheckpoint(os.path.join(self.output_dir, 'model-{epoch:02d}.hdf5'), save_weights_only=True)
        early_stop = EarlyStopping(patience=3)
        train_history = self.model.fit(self.dataset.x_train, self.dataset.y_train,
                                       validation_data=(self.dataset.x_val, self.dataset.y_val),
                                       epochs=100, batch_size=self.batch_size,
                                       class_weight={0: 1, 1: self.dataset.pos_weight},
                                       callbacks=[logger, saver, early_stop])

        return train_history

    def save_test_predictions(self):
        print('running predictions on test set')
        predictions = self.model.predict(self.dataset.x_test)
        print('saving predictions to file')
        test_idx_out = self.dataset.test_idx
        test_idx_out.sort()
        pickle.dump((self.age_brackets, predictions, self.dataset.y_test, test_idx_out),
                    open(os.path.join(self.output_dir, 'test_predictions.pkl'), 'wb'))

    def run_hyperopt(self, params):
        self.model_outputs = []

        self.output_dir = os.path.join(self.base_output_dir, 'trial_{}'.format(params['trial_nr']))
        if not os.path.isdir(self.output_dir):
            os.makedirs(self.output_dir)
        self.nr_lstm_units = params['nr_lstm_units']
        self.so_embedding_sizes = params['nr_emb_units']
        self.batch_size = params['batch_size']
        self.yearly_inputs, self.so_inputs = self.create_model_inputs()
        self.model = self.create_model()
        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        train_history = self.train()
        self.save_test_predictions()
        val_loss = train_history.history['val_loss']
        return min(val_loss)


if __name__ == '__main__':

    configuration = configparser.ConfigParser()
    configuration.read("config.ini")
    preproc_data = SunyRnnPreproc(configuration)
    study = SunyRnnModel(configuration, preproc_data)

    study.model.compile(optimizer='adam', loss='binary_crossentropy')
    history = study.train()
    study.save_test_predictions()

    # example code to read in earlier stored predictions
    brackets, preds, truth, test_idx = pickle.load(open(os.path.join(study.base_output_dir, 'test_predictions.pkl'), 'rb'))

    # example code to run predictions for new test subjects
    weights_file = ''
    configuration = configparser.ConfigParser()
    configuration.read("config.ini")
    preproc_data = SunyRnnPreproc(configuration)
    study = SunyRnnModel(configuration, preproc_data)
    study.model.load_weights(weights_file)
    study.save_test_predictions()
