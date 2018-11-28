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




def retrieve_coordinates(image):
  rowNums =[]
  colNums =[]
  for index in range(image.shape[0]):
    if(image[index, :].sum()>0):
      rowNums.append(index)
    if(image[:, index].sum()>0):
      colNums.append(index)
  return (np.min(rowNums), np.max(rowNums), np.min(colNums), np.max(colNums))

def resize_dataset(file, test_percent = .33, previous_shape =(160,160), image_size = (40,40), bg_type='0'):
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
        im = d[0][0].toarray().reshape(previous_shape)
        box = retrieve_coordinates(im)
        im = im[box[0]:box[1], box[2]:box[3]]
        if im.shape[0]< 3 or im.shape[1] < 3:
            continue
        im = resize(im, image_size, mode ='constant', preserve_range=True)
        if bg_type  == "-1":
            im[im==im.min()] = -1
            # plt.imshow(im)
            # plt.show()
            # quit()
        elif bg_type  == "scaled_negative":
            im[im==im.min()] = -1
            im += 1
            im = Normalizer().fit_transform(im)
            # plt.imshow(im)
            # plt.show()
            # quit()

        row = np.append(im.ravel(), [d[0][1], d[1], serial_slice_num])
        dataset.append(row)
    dataset = np.array(dataset)
    dataset = csr_matrix(dataset)
    if bg_type  == "-1":
        print ("-1 Data save")
        save_npz(save_location+"negative_backgrounds/"+filename+'_'+str(image_size[0])+'_'+str(image_size[1]), dataset)
    elif bg_type  == "scaled_negative":
        print ("scaled negative save")
        save_npz(save_location+"scaled_negative/"+filename+'_'+str(image_size[0])+'_'+str(image_size[1]), dataset)
    elif bg_type  == "0":
        print ("Zero data save")
        save_npz(save_location+"zero/"+filename+'_'+str(image_size[0])+'_'+str(image_size[1]), dataset)

save_location ='./dataset/cropped/'
serial_slice_num = 0
all_files = glob.glob('with_zeros/*.npy')
all_files = sorted(all_files)
print (len(all_files))
for afile in all_files:
  resize_dataset(afile,  bg_type  = "scaled_negative")
