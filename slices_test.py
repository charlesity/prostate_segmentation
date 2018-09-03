'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import matplotlib.pyplot as plt
import numpy as np
import os
s = np.load('./Processed_data/slices/pslice_0.npy')
print (s[:, 786])
