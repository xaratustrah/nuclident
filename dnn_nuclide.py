#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Deep learning with TensorFlow for identification of nuclidic species in heavy ion storage rings based on atomic mass data base.


2017 Xaratustrah

'''

__version__ = '0.0.1'

import os
# turn off debug warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import argparse
import sys
import logging as log

from particle import *
from ui_interface import *
from amedata import *
from ring import Ring

import tflearn
import numpy as np
import h5py
import pickle


class NeuralNetwork:
    def __init__(self, file_basename):
        self.file_basename = file_basename

    def save_data_to_file(self):
        print('Saving', self.n_rows, 'species into HDF5...')
        # write HDF5
        with h5py.File('{}.h5'.format(self.file_basename), 'w') as hf:
            hf.create_dataset(self.file_basename,  data=self.nuclidic_data)

        with open('{}.pik'.format(self.file_basename), 'wb') as fp:
            pickle.dump(self.nuclidic_labels, fp)

    def load_data_from_file(self):
        print('Reading data...')
        with h5py.File('{}.h5'.format(self.file_basename), 'r') as hf:
            self.nuclidic_data = hf[self.file_basename][:]
        with open('{}.pik'.format(self.file_basename), 'rb') as fp:
            self.nuclidic_labels = pickle.load(fp)

    def save_model_to_file(self):
        print('Saving model to file...')
        self.model.save('{}.tfl'.format(self.file_basename))

    def load_model_from_file(self):
        print('Loading model from file...')
        self.model.load('{}.tfl'.format(self.file_basename))

        # ------

    def define_net(self, in_node, out_node, intermediate_node=10):
        # import tflearn; tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, in_node])
        net = tflearn.fully_connected(net, intermediate_node)
        net = tflearn.fully_connected(net, intermediate_node)
        # no of output nodes = no of Y classes
        net = tflearn.fully_connected(net, out_node, activation='linear')
        net = tflearn.regression(net)

        # Define model
        self.model = tflearn.DNN(net)

    def prepare(self):
        ame_data = AMEData(DummyIFace())

        # create reference particle
        p = Particle(6, 6, ame_data, Ring('ESR', 108.5))
        p.qq = 3
        p.ke_u = 422
        p.path_length_m = 108.5
        p.f_analysis_mhz = 245
        p.i_beam_uA = 1.2
        print('Reference particle:', p)
        print('Isobars:')
        for pp in p.get_isobars():
            print(pp)
        # get some nuclides
        # nuclides = p.get_all_in_all()
        # nuclides = p.get_nuclides(57, 59, 81, 83, 2)
        nuclides = p.get_nuclides(20, 92, 40, 143, 10)
        self.n_rows = int(len(nuclides))

        self.nuclidic_data = np.array([])
        self.nuclidic_labels = []

        for pp in nuclides:
            pp.calculate_revolution_frequency()
            brho = pp.get_magnetic_rigidity()
            values = [pp.revolution_frequency, brho]
            self.n_cols = len(values)

            self.nuclidic_data = np.append(
                self.nuclidic_data, values)
            self.nuclidic_labels.append(pp.get_short_name())

        # print(self.nuclidic_labels)
        self.nuclidic_data = np.reshape(
            self.nuclidic_data, (self.n_rows, 2))

    def train(self):
        self.n_rows = len(self.nuclidic_labels)

        one_hot = np.identity(self.n_rows)

        # print(one_hot)
        # print(self.nuclidic_data)
        # print(self.nuclidic_labels)
        # print(self.n_rows)
        # print(self.n_cols)

        # Create the network model
        self.define_net(self.n_cols, self.n_rows)

        # Start training (apply gradient descent algorithm)
        self.model.fit(self.nuclidic_data, one_hot, n_epoch=3,
                       validation_set=0.1,
                       batch_size=4, show_metric=True)

    def predict(self):
        dut = np.array([7.98469972, 0.])
        pred = self.model.predict([dut])
        print(pred)


# ===============

if __name__ == '__main__':
    scriptname = 'dnn_nuclide'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--prepare", help="Prepare the network", action="store_true")
    parser.add_argument(
        "-t", "--train", help="Train neural network.", action="store_true")
    parser.add_argument(
        "-d", "--predict", help="Predict", action="store_true")
    parser.add_argument(
        "-a", "--all", help="Do it all", action="store_true")

    args = parser.parse_args()

    print('{} {}'.format(scriptname, __version__))

    dnn = NeuralNetwork('dnn_nuclide')

    if args.prepare:
        dnn.prepare()
        dnn.save_data_to_file()

    elif args.train:
        dnn.load_data_from_file()
        dnn.train()
        dnn.save_model_to_file()

    elif args.predict:
        dnn.load_model_from_file()
        dnn.predict()

    elif args.all:
        dnn.prepare()
        dnn.train()
        dnn.predict()
    else:
        print('Nothing to do.')
        sys.exit()
