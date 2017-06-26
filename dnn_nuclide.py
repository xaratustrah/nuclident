#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Deep learning with TensorFlow for identification of nuclidic species in heavy ion storage rings based on atomic mass data base.


2017 Xaratustrah

'''


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

__version__ = '0.0.1'


def prepare():
    ame_data = AMEData(DummyIFace())
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
    some_nuclides = p.get_nuclides(20, 92, 40, 143, 10)
    nuclidic_data = np.array([])
    for pp in some_nuclides:
        nuclidic_data = np.append(
            nuclidic_data, [pp.tbl_zz, pp.tbl_nn, pp.qq, pp.get_ionic_moq()])
        nuclidic_data = np.reshape(
            nuclidic_data, (int(len(nuclidic_data) / 4), 4))

    print('Saving', len(nuclidic_data), 'species into HDF5...')
    # write HDF5
    with h5py.File('moverq.h5', 'w') as hf:
        hf.create_dataset('moverq',  data=nuclidic_data)
    print('Done.')


def train():
    # Read HDF5
    print('Reading data...')
    with h5py.File('moverq.h5', 'r') as hf:
        d = hf['moverq'][:]

    labels = np.transpose(np.array([d[:, 3]]))

    # import tflearn; tf.reset_default_graph()
    # Build neural network
    net = tflearn.input_data(shape=[None, 4])
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 10)
    net = tflearn.fully_connected(net, 1, activation='linear')
    net = tflearn.regression(net)

    # Define model
    model = tflearn.DNN(net)
    # Start training (apply gradient descent algorithm)
    model.fit(d, labels, n_epoch=10, batch_size=100, show_metric=True)
    # Manually save model
    model.save('moverq.tfl')

# ------------


if __name__ == '__main__':
    scriptname = 'dnn_nuclide'
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t", "--train", help="Train neural network.", action="store_true")
    parser.add_argument(
        "-p", "--prepare", help="Prepare the network", action="store_true")

    args = parser.parse_args()

    print('{} {}'.format(scriptname, __version__))

    if args.prepare:
        prepare()
    elif args.train:
        train()
    else:
        print('Nothing to do.')
        sys.exit()
