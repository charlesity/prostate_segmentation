from scipy.sparse import csr_matrix #sparse matrix
import seaborn as sbn  #heatmap plot library
import os
import numpy as np
import scipy as sp
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt

import pickle

#libact libraries
from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from

import copy
from libact.base.dataset import Dataset
from libact.models import LogisticRegression, SVM
from GaussianNaiveBayes import GaussianNaiveBayes
from libact.query_strategies import UncertaintySampling
try:
    from sklearn.model_selection import train_test_split
except:
    from sklearn.cross_validation import train_test_split

from libact.labelers import IdealLabeler

from segmentation_utilities import *





import nrrd #library for reading nrrd files
import pydicom as pydicom # library for reading dicom files

import glob as glob  # for recursively walking through the os and selected files

from skimage.color import label2rgb 

#spectral clustering libraries
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from skimage.segmentation import mark_boundaries


np.set_printoptions(threshold=np.nan)




"""
Define a custom labeler based on the nrrd files
"""
import numpy as np
import pickle

from libact.base.interfaces import Labeler
from libact.utils import inherit_docstring_from
from matplotlib import pyplot


class DicomLabeler(Labeler):
    """
    Provide the errorless/noiseless label to any feature vectors being queried.
    Parameters
    ----------
    dataset: Dataset object
        Dataset object with the ground-truth label for each sample.
    """

    def __init__(self, **kwargs):
        self.label_name = kwargs.pop('label_name', None)
        
        self.ax = kwargs.pop('mainDisplay', None)
        # if self.ax == None:
        #     raise ValueError('Ax must not be null')
        
            
        self.size_dataset = kwargs.pop('size_dataset', None)
        if self.size_dataset is None:
            raise ValueError('Size of dataset must be specified')
        
        # self.range_box = np.array([[i, i+self.N_clusters-1] for i in range(0, self.size_dataset, self.N_clusters)])
    
            
        
            
    @inherit_docstring_from(Labeler)
    def label(self, feature, ask_id):        
        theImage_label, theImage = reconstruct_labels(ask_id, ds)
        
        # self.ax[0].imshow(mark_boundaries(theImage, theImage_label))
        self.ax[0, 0].imshow(theImage, cmap = 'gray')
        
        self.ax[0, 1].imshow(label2rgb(theImage_label, theImage))
        
        # plot contours arround segmented images
        

        # plot only the segmented areas in the right axis
        self.ax[1,0].imshow(feature.reshape(160,160), cmap ='gray')


        # plt.imshow(feature.reshape(160,160))
        
        plt.draw()
        
        banner = "Enter the associated label (0 for background and 1 for foreground) or type stop to quit labeling"
        

        if self.label_name is not None:
            banner +=str(self.label_name)
        lbl = input(banner)
        
        while (self.label_name is not None) and lbl not in self.label_name:
            print ("Invalid label, re-enter the associated label")
            lbl = input(banner)
        
        return self.label_name.index(lbl)


dicom_training_folder= "Training/DOI/Prostate3T-01-0001/"

dicomSet = glob.glob(dicom_training_folder+"**/*.dcm", recursive=True)
dicomSet = sorted(dicomSet)

i = 5
theLabels, theImage = get_watershed_superpixels(dicomSet[i])

ds = []
shape =(160,160)
    
for l in np.unique(theLabels):
    aMask = np.zeros(shape)   #each mask task the shape of the image
    aMask[theLabels==l] = theImage[theLabels==l]    
            
    atuple = (csr_matrix(aMask), i)
    ds.append(atuple) #compress the arrays as sparse column matrix and append to the list

ds = np.array(ds)



y_index = np.arange(len(ds))



ds_sliced = ds[:, 0]


theDs = np.empty((len(ds_sliced), 160**2))
for i, d in enumerate(ds_sliced):
    theDs[i]= (d.todense().flatten())    
n_labeled = int(.01* len(theDs))    
   
ask_id = -1
#start with a randomly labeled dataset
y_random = np.random.randint(2, size=len(ds_sliced))




#labels = np.concatenate([y_random[:n_labeled], [None] * (len(y_random) - n_labeled)])
labels = np.concatenate([[0, 1], [None] * (len(y_random) - 2)])  #label only four samples
# labels = np.array([None]*len(ds_sliced))


trn_ds = Dataset(theDs, labels)
X, _ = zip(*trn_ds.data)


# # points_ = [image_location(i, (160,160), X) for i in range(len(X))]
# # X_values = [p[0] for p in points_ ]
# # Y_values = [p[1] for p in points_ ]

# # plt.scatter(X_values, Y_values)
# # plt.show()

# lb, pic = reconstruct_labels(80)


# im = mark_boundaries(pic, lb)
# # plt.imshow(pic)
# # plt.figure()
# # plt.imshow(im)

# # plt.show()

# plt.imshow(label2rgb(lb, pic))
# plt.show()
qs = UncertaintySampling(trn_ds, method='lc', model = GaussianNaiveBayes())
model = GaussianNaiveBayes()

n_classes = 2

fig, axes = plt.subplots(2, 2)
# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# axes.set_title('Dicom Images')
if ask_id==-1:
    axes[0,0].imshow(np.zeros((160,160)))
else:
     X, _ = zip(*trn_ds.data)
     
     axes[0,0].imshow(reconstruct_image(ask_id, ds), cmap='gray')
        

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
               shadow=True, ncol=5)
plt.show(block=False)




# lbr = DicomLabeler(label_name=[str(lbl) for lbl in range(n_classes)], mainDisplay = (ax1, ax2), size_dataset = len(theDs))
lbr = DicomLabeler(label_name=[str(lbl) for lbl in range(n_classes)], mainDisplay = axes, size_dataset = len(theDs))


quota = len(y_random) - n_labeled
label_count = 0
for _ in range(quota):

          
    # Standard usage of libact objects
    try:        
        ask_id = qs.make_query()
        print ('Ask ID', ask_id)
        
        X, y = zip(*trn_ds.data)
        lb = lbr.label(X[ask_id], ask_id)
        
        trn_ds.update(ask_id, lb)

        model.train(trn_ds)

        X, y = zip(*trn_ds.data)
        
        
        
        
        plt.draw()

        if label_count%10 == 0:
            quit_signal = input("Type quit to stop labeling or enter to continue ")
            if quit_signal.lower() == "quit" :
                break
        label_count = label_count + 1
    except ValueError as valerr:
        print (valerr)
        break


#test the segmentation

dicom_training_folder= "Training/DOI/Prostate3T-01-0003/"

dicomSet = glob.glob(dicom_training_folder+"**/*.dcm", recursive=True)
dicomSet = sorted(dicomSet)

i = 5

theLabels, theImage = get_watershed_superpixels(dicomSet[i])



shape = (160,160)
for l in np.unique(theLabels):
    aMask = np.zeros(shape)   #each mask task the shape of the image
    aMask[theLabels==l] = theImage[theLabels==l]
    #predict based on the appearance models
    theLabels[theLabels==l] = model.predict([aMask.flatten()])[0]    
#   super_pixel_list.append([csr_matrix(aMask), i]) #compress the arrays as sparse column matrix and append to the list
            


fig2, axes = plt.subplots(1, 2)

axes[0].imshow(label2rgb(theLabels, theImage), cmap='gray')
axes[1].imshow(theImage, cmap='gray')


plt.show()




    
