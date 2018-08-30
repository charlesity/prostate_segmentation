from scipy.sparse import csr_matrix #sparse matrix
import os
import numpy as np
import scipy as sp
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt

import copy

from segmentation_utilities import *





import nrrd as nr #library for reading nrrd files
import pydicom as pydicom # library for reading dicom files

import glob as glob  # for recursively walking through the os and selected files

from skimage.color import label2rgb 

#spectral clustering libraries
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
from skimage.segmentation import mark_boundaries



def generate_and_label(algorithm):
    dicom_training_folder= "Training/DOI/"

    just_folders = sorted(glob.glob("Training/DOI/*/"))
    nrrd_source = "Training_nrrd/"

    fig, axes = plt.subplots(2, 2, figsize=(40,40))

    # f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    # axes.set_title('Dicom Images')

    axes[0,0].imshow(np.zeros((160,160)))
    axes[0,1].imshow(np.zeros((160,160)))
    axes[1,0].imshow(np.zeros((160,160)))
    axes[1,1].imshow(np.zeros((160,160)))
            

    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True,
                   shadow=True, ncol=5)
    plt.show(block=False)

    shape =(160,160)
    for folder in just_folders:
        dicomSet = glob.glob(folder+"/**/*.dcm", recursive=True)
        dicomSet = sorted(dicomSet)
        folderName = str(folder[13:-1])

        if os.path.isfile(folderName+'_'+algorithm[0]+'.npy'):
            continue    #already labeled skip to next file

        NRRDFile = str(folder[13:-1])+".nrrd"    #pick the associated nrrd file
        fig.suptitle('Images in '+folderName+ ' Folder', fontsize=20)
        #set figure title

        nrrd_data, header = nr.read(nrrd_source+NRRDFile)
        
        if (nrrd_data.shape[0] != 320):
            nrrd_data = resize(nrrd_data, (320, 320))
        nrrd_data = rescale (nrrd_data, .5)

        superpixelset = []
        for fileNo, dicomFile in enumerate(dicomSet):            
            theLabels, theImage = algorithm[1](dicomFile)  #algorithm[1] is a function
            # theLabels, theImage = get_watershed_superpixels(dicomFile)


            #nrrd for display to guide manual segmentation

            nrrd_data_segmented = nrrd_data[:, :, fileNo].T #retrieve and transpose data
            xMaskNRRD = np.zeros(shape)
            xMaskNRRD[nrrd_data_segmented == nrrd_data_segmented.max()] = theImage[nrrd_data_segmented == nrrd_data_segmented.max()] #extract segmented


            if nrrd_data_segmented.max() == 0:
                print ("Thats the maximum")

            axes[1,0].imshow(xMaskNRRD)
            axes[1,0].set_title('Segmented From Source (For guidance')
            axes[1,0].set_axis_off()          


            
            axes[0,0].imshow(theImage)
            axes[0,0].set_title('Original Image with File Number '+ str(fileNo))
            axes[0,0].set_axis_off()
            axes[0,1].imshow(mark_boundaries(theImage, theLabels))
            axes[0,1].set_title('Segmented and marked with File Number '+str(fileNo))
            axes[0,1].set_axis_off()
            
            axes[1,0].imshow(xMaskNRRD)
            axes[1,0].set_title('Original Segmented with File Number '+str(fileNo))
            axes[1,0].set_axis_off()          

            banner = "Enter (0 for background, 1 for foreground 'all_zero' to set all background or 'rem' to label remaining zero) or stop to quit \n"
            
            lb = None
            os.system('clear')    
            for l in np.unique(theLabels):
                # os.system('clear')
                plt.axis('off')
                aMask = np.zeros(shape)   #each mask task the shape of the image
                aMask[theLabels==l] = theImage[theLabels==l]

                if lb =="all_zero":
                    continue
                elif lb == "rem":
                    atuple = ([csr_matrix(aMask.flatten()), lb], fileNo)
                    superpixelset.append(atuple) #compress the arrays as sparse column matrix and append to the list
                elif lb == "stop":
                    quit();
                else:
                    #segmented part
                    axes[1,1].imshow(aMask)
                    axes[1,1].set_title('Segmented to be labeled')
                    axes[1,1].set_axis_off()
                    plt.draw()  #redraw the plots
                    lb = input(banner)
                 
                    while (lb not in ['0','1', 'all_zero', 'stop', 'rem']):    
                        print ("Invalid label,you entered {} re-enter the associated label ".format(lb))
                        lb = input(banner)

                    atuple = ([csr_matrix(aMask.flatten()), lb], fileNo)
                    superpixelset.append(atuple) #compress the arrays as sparse column matrix and append to the list

        theSuperPixels = np.asarray(superpixelset)
        np.save(folderName+'_'+algorithm[0], theSuperPixels)   
        
algorithms =[('WaterShed', get_watershed_superpixels)]
# algorithms =[('WaterShed', get_watershed_superpixels), ('SlIC', get_slic_superpixels)]


for a in algorithms:
    generate_and_label(a)
