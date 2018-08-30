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
    rr = np.where(image[index, :] >= 0)
    cc = np.where(image[:, index] >= 0)
    if rr[0].shape[0] > 0:
      rowNums.append(index)
    if cc[0].shape[0] > 0:
      colNums.append(index)
  # print (np.min(rowNums), np.max(rowNums), np.min(colNums), np.max(colNums))
  return (np.min(rowNums), np.max(rowNums), np.min(colNums), np.max(colNums))

def resize_dataset(file, test_percent = .33, previous_shape =(160,160), image_size = (28,28)):
  filename, _ = os.path.splitext(os.path.basename(file))
  #if its already generated skip it
  # if os.path.isfile(filename+'_'+str(image_size[0])+'_'+str(image_size[1])+'.npy'):
  #   return
  print ('Processing File ', filename)
  arr = np.load(file)
  dataset = list()
  for d in arr:
    im =d[0][0].toarray().reshape(previous_shape)
    im[im==0.0] = None
    box = retrieve_coordinates(im)    
    im = im[box[0]:box[1], box[2]:box[3]]
    if im.shape[0] < 2 or im.shape[1] <2:
      continue
    im = resize(im, image_size)
    im[np.isnan(im)] = -1
    # print (im)
    # plt.imshow(im)
    # plt.show()
    # quit()
    row = np.append(im.ravel(), [d[0][1], d[1]])
    dataset.append(row)
  np.save(location+filename+'_'+str(image_size[0])+'_'+str(image_size[1])+"_with_minus_1", dataset)

location = "./rescaled_minus/"
all_files = glob.glob('../with_zeros/*.npy')
for afile in all_files:
  resize_dataset(afile)
  # break
# arr = np.load(all_files[0])
# database =np.array([np.concatenate((d[0][0].toarray().ravel(), [d[0][1]], [d[1]])) for d in arr])

# im =database[450, :25600].reshape(160,160)
# im = resize(im, (30,30))
# plt.imshow(im)
# plt.show()