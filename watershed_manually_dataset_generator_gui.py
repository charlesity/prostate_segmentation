from six.moves import tkinter as Tk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2TkAgg)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from scipy.sparse import csr_matrix #sparse matrix
import os
import numpy as np
import scipy as sp
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib
matplotlib.use('GTKAgg') 
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



def displayAndLabel():
    global current_image_labels, current_image
    NRRDFile = str(folder[13:-1])+".nrrd"    #pick the associated nrrd file
    fig.suptitle('Images in '+folderName+ ' Folder', fontsize=20)
    #set figure title

    nrrd_data, header = nr.read(nrrd_source+NRRDFile)

    if (nrrd_data.shape[0] != 320):
        nrrd_data = resize(nrrd_data, (320, 320))
    nrrd_data = rescale (nrrd_data, .5)

    superpixelset = []
            
    #nrrd for display to guide manual segmentation

    nrrd_data_segmented = nrrd_data[:, :, counter].T #retrieve and transpose data
    xMaskNRRD = np.zeros(shape)
    xMaskNRRD[nrrd_data_segmented > nrrd_data_segmented.mean()] = current_image[nrrd_data_segmented > nrrd_data_segmented.mean()] #extract segmented

    axes[0,0].imshow(current_image)
    axes[0,0].set_title('Original Image with File Number '+ str(counter))


    axes[0,1].imshow(mark_boundaries(current_image, current_image_labels))
    axes[0,1].set_title('Segmented and marked with File Number '+str(counter))



    axes[1,0].imshow(xMaskNRRD)
    axes[1,0].set_title('Segmented From Source (For guide)')

    global canvas
    canvas.draw()


def nextIamgeInFolder():
    global counter, fileSuperpixels    
    counter+=1
    for su in fileSuperpixels:
        #check if it has been added already
        added_already = False
        for d in dataset:
            if (d[0][0].todense().reshape(shape)== su).all():
                added_already = True                
                break
        if added_already:
            continue
        else:
            #add to the list and label zero
            atuple = ([csr_matrix(su.flatten()), 0], counter)
            dataset.append(atuple)

    axes[1,1].imshow(np.zeros(shape)+255)
    #get all superpixels and label them zero


    if counter < number_files:        
        setSuperPixelSet()
        displayAndLabel()
    else:
        #save the folder with superpixels
        np.save(folderName+'_manual_gui'+'_watershed', dataset)        
        check() 
        displayAndLabel()

    


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

def onClick(event):    
    global dataset, counter, fileSuperpixels
    theSuperPixelSelected = selectSuperPixel(event.xdata, event.ydata)
    if theSuperPixelSelected is not None:
        if event.button == 1:
            # add to the superpixel list
            atuple = ([csr_matrix(theSuperPixelSelected.flatten()), 1], counter)
            axes[1,1].imshow(theSuperPixelSelected)
            dataset.append(atuple)

        #right click button, then set to zero label
        elif event.button == 3:
            for index, s in enumerate(fileSuperpixels):
                if (s == theSuperPixelSelected).all():
                    #counter is the file_number within folder
                    fileSuperpixels[index] = ([csr_matrix(theSuperPixelSelected.flatten()), 0], counter)   #set to zero
                    axes[1,1].imshow(np.zeros(shape)+255)

    canvas.draw()

def selectSuperPixel(xPoint, yPoint):    
    global fileSuperpixels
    for s in fileSuperpixels:
        if s[yPoint.astype(np.int)][xPoint.astype(np.int)]!= 0: # x-axis is y and y-axis is x in GUI geometry
            return s
    return None

def setSuperPixelSet():
    global  current_image_labels, current_image, fileSuperpixels
    fileSuperpixels = []
    current_image_labels, current_image = get_watershed_superpixels(dicomSet[counter])
    #ID of the superpixel within the image, and generated l is the label of  image
    for l in np.unique(current_image_labels):
        aMask = np.zeros(current_image.shape)
        aMask[current_image_labels==l] = current_image[current_image_labels==l]
        fileSuperpixels.append(aMask)
            # if aMask[yPoint.astype(np.int)][xPoint.astype(np.int)]!= 0: # x-axis is y and y-axis is x in GUI geometry
            #     return aMask

def check():
    global folderName, counter, number_files, dicomSet, folder
    for f in just_folders:
        if os.path.isfile(f[13:-1]+'_manual_gui'+'_watershed'+'.npy'):
            continue
        else:                        
            folder = f
            counter = 0
            dicomSet = glob.glob(folder+"/**/*.dcm", recursive=True)
            dicomSet = sorted(dicomSet)
            number_files = len(dicomSet)
            folderName = str(f[13:-1])
            setSuperPixelSet()            
            return
    quit()


root = Tk.Tk()
root.wm_title("Prostate Manual Labeler")


dicom_training_folder= "Training/DOI/"

just_folders = sorted(glob.glob("Training/DOI/*/"))
nrrd_source = "Training_nrrd/"
        
shape =(160,160)
dicomSet = None
folderName = None
number_files = None
current_image_labels, current_image = None, None
counter = 0
folder = None
fileSuperpixels = None

fig, axes = plt.subplots(2, 2)
canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)





# toolbar = NavigationToolbar2TkAgg(canvas, root)
# toolbar.update()
# canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

button = Tk.Button(master=root, text="Quit", command=_quit)
button2 = Tk.Button(master=root, text="Next Image", command=nextIamgeInFolder)
button.pack()
button2.pack()
canvas.mpl_connect("button_press_event", onClick)

dataset = []

check()
displayAndLabel()
Tk.mainloop()