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


def define_net():
    # import tflearn; tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, 2])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    # no of output nodes = no of Y classes
    net = tflearn.fully_connected(net, 3, activation='softmax')
    net = tflearn.regression(net)

    # Define model
    return tflearn.DNN(net)


def prepare():
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
    #nuclides = p.get_all_in_all()
    #nuclides = p.get_nuclides(57, 59, 81, 83, 3)
    nuclides = p.get_nuclides(20, 92, 40, 143, 10)
    nuclidic_data = np.array([])
    for pp in nuclides:
        nuclidic_data = np.append(
            nuclidic_data, [pp.tbl_zz, pp.tbl_nn, pp.qq, pp.get_magnetic_rigidity(), p.revolution_frequency])
        nuclidic_data = np.reshape(
            nuclidic_data, (int(len(nuclidic_data) / 5), 5))

    print('Saving', len(nuclidic_data), 'species into HDF5...')
    # write HDF5
    with h5py.File('nuclides.h5', 'w') as hf:
        hf.create_dataset('nuclides',  data=nuclidic_data)
    print('Done.')


def train():
    # Read HDF5
    print('Reading data...')
    with h5py.File('nuclides.h5', 'r') as hf:
        d = hf['nuclides'][:]
    data_part = d[:, 3:]
    labels = d[:, 0:3]
    model = define_net()

    # Start training (apply gradient descent algorithm)
    model.fit(data_part, labels, n_epoch=10, batch_size=4, show_metric=True)
    # Manually save model
    model.save('nuclides.tfl')


def evaluate():
    model = define_net()
    try:
        model.load('nuclides.tfl')
        print('Loaded!')
    except:
        pass

    dut = np.array([8.0422383, 0.])
    pred = model.predict([dut])
    print(pred)


# ------------
if __name__ == '__main__':
    scriptname = 'dnn_nuclide'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--prepare", help="Prepare the network", action="store_true")
    parser.add_argument(
        "-t", "--train", help="Train neural network.", action="store_true")
    parser.add_argument(
        "-e", "--evaluate", help="Evaluate", action="store_true")

    args = parser.parse_args()

    print('{} {}'.format(scriptname, __version__))

    if args.prepare:
        prepare()
    elif args.train:
        train()
    elif args.evaluate:
        evaluate()
    else:
        print('Nothing to do.')
        sys.exit()
