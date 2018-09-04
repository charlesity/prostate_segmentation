'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''
import matplotlib.pyplot as plt
from skimage.transform import resize

import os
import glob as glob
import numpy as np
from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix, save_npz

def resize_dataset(file, test_percent = .33, previous_shape =(160,160), image_size = (60,60)):
    global serial_slice_num
    filename, _ = os.path.splitext(os.path.basename(file))
    # if its already generated skip it
    # if os.path.isfile(filename+'_'+str(image_size[0])+'_'+str(image_size[1])+'.npy'):
    #   return
    print ('Processing File ', filename)
    arr = np.load(file)
    dataset = list()
    current_no = 0
    for d in arr:
        if (d[1] > current_no):
          current_no += 1
          serial_slice_num = serial_slice_num +1
    # print (d[1], serial_slice_num)
        im = d[0][0].toarray().reshape(previous_shape)
        im[im==im.min()] = None
        im = resize(im, image_size, mode ='constant', preserve_range=True)
        im[np.isnan(im)] = -1
        im = im + 1
        im = Normalizer().fit_transform(im)
        # plt.imshow(im)
        # plt.show()pyth
        row = np.append(im.ravel(), [d[0][1], d[1], serial_slice_num])
        dataset.append(row)
    dataset = np.array(dataset)
    dataset = csr_matrix(dataset)
    save_npz(save_location+filename+'_'+str(image_size[0])+'_'+str(image_size[1]), dataset)
    # np.save(save_location+filename+'_'+str(image_size[0])+'_'+str(image_size[1]), dataset)

serial_slice_num = 0
save_location = './dataset/'
all_files = glob.glob('with_zeros/*.npy')
all_files = sorted(all_files)
print (len(all_files))
for afile in all_files:
  resize_dataset(afile)
