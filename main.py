#!/usr/bin/env python
import numpy as np
import os
import sys

from Network import Network
from data_handling import load_training_settings, load_testing_settings, load_random_data_set

rng = np.random.RandomState(42)

if len(sys.argv) != 2:
    print("This script requires 1 parameter: filename to save results to.");
    exit(-1)

results_filename = sys.argv[1];

# define inputs and filters
batchsize     = 5
in_channels   = 1
in_time       = 16
in_width      = 170
in_height     = 128
flt_channels  = [64, 128, 256, 256, 256]
flt_time      = [1, 2, 2, 2, 1]
flt_width     = [3, 4, 5, 7, 2]
flt_height    = [3, 4, 5, 7, 2]
learning_rate = .1
num_epochs 	  = 5

training_data_settings = load_training_settings();
testing_data_settings = load_testing_settings();
dnn = Network((batchsize, in_time, in_channels, in_height, in_width), 
               [(flt_channels[0], flt_time[0], in_channels, flt_height[0], flt_width[0]), 
                (flt_channels[1], flt_time[1], flt_channels[0], flt_height[1], flt_width[1]),
                (flt_channels[2], flt_time[2], flt_channels[1], flt_height[2], flt_width[2]),
                (flt_channels[3], flt_time[3], flt_channels[2], flt_height[3], flt_width[3]),
                (flt_channels[4], flt_time[4], flt_channels[3], flt_height[4], flt_width[4])
                ],
               [(1,2,2), (1,2,2), (2,2,2), (2,2,2), (2,2,2)], 
               6, learning_rate, rng);

target = open(results_filename, 'w')
for round in range(0, num_epochs):
    print("Round %d" % round)
    (train_y, train_x) = load_random_data_set(training_data_settings, batchsize, in_time, in_channels, in_width, in_height, round=('training%d' % round))
    (test_y, test_x) = load_random_data_set(testing_data_settings, batchsize, in_time, in_channels, in_width, in_height, round=('testing%d' % round))
    print(dnn.train(train_x, train_y))
    result = dnn.validate(test_x, test_y)
    print(result)
    target.write("%f\n" % result)
target.close()
