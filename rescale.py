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



def retrieve_coordinates(image):
  rowNums =[]
  colNums =[]
  for index in range(image.shape[0]):
    if(image[index, :].sum()>0):
      rowNums.append(index)
    if(image[:, index].sum()>0):
      colNums.append(index)
  return (np.min(rowNums), np.max(rowNums), np.min(colNums), np.max(colNums))

def resize_dataset(file, test_percent = .33, previous_shape =(160,160), image_size = (28,28)):
  global serial_slice_num

  

  filename, _ = os.path.splitext(os.path.basename(file))
  if its already generated skip it
  if os.path.isfile(filename+'_'+str(image_size[0])+'_'+str(image_size[1])+'.npy'):
    return
  print ('Processing File ', filename)
  arr = np.load(file)
  dataset = list()
  current_no = 0
  for d in arr:
    if (d[1] > current_no):
      current_no += 1
      serial_slice_num = serial_slice_num +1
    # print (d[1], serial_slice_num)
    im =d[0][0].toarray().reshape(previous_shape)
    box = retrieve_coordinates(im)
    im = im[box[0]:box[1], box[2]:box[3]]
    if im.shape[0] < 2 or im.shape[1] <2:
      continue
    im = resize(im, image_size)
    row = np.append(im.ravel(), [d[0][1], d[1], serial_slice_num])
    dataset.append(row)
  np.save(filename+'_'+str(image_size[0])+'_'+str(image_size[1]), dataset)


serial_slice_num = 0
all_files = glob.glob('with_zeros/*.npy')
all_files = sorted(all_files)
print (len(all_files))
for afile in all_files:
  resize_dataset(afile)
