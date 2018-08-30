from scipy.sparse import csr_matrix #sparse matrix
import seaborn as sbn  #heatmap plot library
import os
import numpy as np
import scipy as sp
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
from skimage import img_as_float, filters as filters, feature
from scipy import ndimage as ndi
from skimage.segmentation import mark_boundaries, slic
from skimage import color, filters


import nrrd #library for reading nrrd files
import pydicom as pydicom # library for reading dicom files

import glob as glob  # for recursively walking through the os and selected files

#watershed libraries
from skimage.morphology import watershed


def get_slic_superpixels(image_file, cluster = 60, sigValue = 2, 
                         compactness = .001, rescale_value = .5, enforce_connectivity=False ):
    aDicomFile = pydicom.read_file(image_file).pixel_array  #read the dicom file
    aDicomFile = img_as_float(aDicomFile)   #convert to floating point
    if (aDicomFile.shape[0] != 320):
        aDicomFile = resize(aDicomFile, (320,320))  # for uniformity
    aDicomFile = rescale(aDicomFile, rescale_value)  #rescale the image
    
    original_File = aDicomFile.copy()
    #enhance the features of the image before generating superpixels
    aDicomFile = filters.gaussian(aDicomFile, sigma=sigValue)  
    
    segments = slic(aDicomFile, n_segments=60, enforce_connectivity=enforce_connectivity, compactness=compactness) + 1    
    
    return segments, original_File  # return label reshaped label array of the clustered regions and the file itself  


def generate_dicom_dataset_slic(source_folder, cluster_size, compactness, cluster_connectivity=False):
    dicomSet = glob.glob(source_folder+"/**/*.dcm", recursive=True)
    dicomSet = sorted(dicomSet)
    
    shape = (160, 160)
    super_pixel_list = []
    
    for i, file in enumerate(dicomSet):
        theLabels, theDicomFile = get_slic_superpixels(file, cluster = cluster_size
                                                       , compactness = compactness, enforce_connectivity=cluster_connectivity)
        for l in np.unique(theLabels):
            aMask = np.zeros(shape)   #each mask task the shape of the image
            aMask[theLabels==l] = theDicomFile[theLabels==l]    
#             super_pixel_list.append([csr_matrix(aMask), i]) #compress the arrays as sparse column matrix and append to the list
                
            atuple = (csr_matrix(aMask), i)
            super_pixel_list.append(atuple) #compress the arrays as sparse column matrix and append to the list
    return super_pixel_list    

def get_spectral_clustering(image_file, label_algorithm, N_Clusters = 30,rescale_value = .5):
    aDicomFile = pydicom.read_file(image_file).pixel_array/255.  #read the dicom file
    if (aDicomFile.shape[0] != 320):
        aDicomFile = resize(aDicomFile, (320,320))  # for uniformity
    aDicomFile = rescale(aDicomFile, rescale_value)  #rescale the image
    graph = image.img_to_graph(aDicomFile)    #build a graph of the image
    labels = spectral_clustering(graph, n_clusters=N_Clusters, assign_labels=label_algorithm, random_state=1) #perform spectral clustering
    return labels.reshape(aDicomFile.shape), aDicomFile  # return label reshaped label array of the clustered regions and the file itself  


def generate_dicom_dataset(source_folder, clusters_per_image = 30, label_algorithm = "kmeans"):
    dicomSet = glob.glob(source_folder+"/**/*.dcm", recursive=True)
    dicomSet = sorted(dicomSet)
    
    shape = (160, 160)
    super_pixel_list = []
    
    for i, files in enumerate(dicomSet):
        theLabels, theDicomFile = get_spectral_clustering(files, label_algorithm = label_algorithm, N_Clusters=clusters_per_image)
        for l in np.unique(theLabels):
            aMask = np.zeros(shape)   #each mask task the shape of the image
            aMask[theLabels==l] = theDicomFile[theLabels==l]    
            super_pixel_list.append(csr_matrix(aMask)) #compress the arrays as sparse column matrix and append to the list
    return super_pixel_list    


def image_location(ask_id, image_shape, X):
    theImage = X[ask_id].reshape(image_shape)
    indexes = (np.where(theImage > theImage.min()))
    return (np.average(indexes[0]), np.average(indexes[1]))

def image_location(theImage):
    indexes = (np.where(theImage > theImage.min()))
    return (np.average(indexes[0]), np.average(indexes[1]))


def reconstruct_labels(ask_id, ds):
    #find the image identification ID in the dataset
    the_image_id = ds[ask_id][1]

    #addup all the superpixels
    image_mask = np.zeros((160,160))
    image_labels = np.zeros((160,160), dtype = int)
    
    indices =np.argwhere(ds[:,1] == the_image_id)
    for i, index in enumerate(indices):
        image_mask += ds[index[0], 0].todense()
        image_labels[ds[index[0], 0].todense() > ds[index[0], 0].todense().min()] = i
    return image_labels ,image_mask


def reconstruct_image(ask_id, ds):
    #find the image identification in the dataset
    the_image_id = ds[ask_id][1]

    #addup all the superpixels
    image_mask = np.zeros((160,160))
    
    indices =np.argwhere(ds[:,1] == the_image_id)
    for i in indices:
        image_mask += ds[i[0], 0].todense()
    return image_mask



def apply_filter(theImage, sigValue=2):
    return filters.gaussian(theImage, sigma=sigValue)    

def find_edges(theImage):
    return filters.sobel(theImage)

def threshold_image (theImage):
    return  filters.threshold_otsu(theImage)


def load_scan(path):
    the_dicom_files = glob.glob(path+"/**/*.dcm", recursive=True)
    slices = [dicom.read_file(s) for s in the_dicom_files]
#     print (slices[0])
    
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices




def get_watershed_superpixels(image_file, sigValue = 2, min_dist =3,rescale_value = .5):
    aDicomFile = pydicom.read_file(image_file).pixel_array  #read the dicom file
    aDicomFile = img_as_float(aDicomFile)   #convert to floating point
    if (aDicomFile.shape[0] != 320):
        aDicomFile = resize(aDicomFile, (320,320))  # for uniformity
    aDicomFile = rescale(aDicomFile, rescale_value)  #rescale the image
    
    original_File = aDicomFile.copy()
    #enhance the features of the image before generating superpixels
    aDicomFile = filters.gaussian(aDicomFile, sigma=sigValue)    
    #find the edges
    aDicomFile = filters.sobel(aDicomFile)
    threshold = filters.threshold_otsu(aDicomFile)
    
#     plt.figure(figsize=(20,20))
#     plt.subplot(2,2,1)
#     plt.imshow(aDicomFile)
    #extract the non-edges to calculate the peeks for watershed algorithm
    non_edges = aDicomFile < threshold
    
#     print (threshold)
#     plt.subplot(2,2,2)
#     plt.imshow(non_edges)
    
    distance_from_edge = ndi.distance_transform_edt(non_edges)
    
    
    
    peeks = feature.peak_local_max(distance_from_edge, min_distance= min_dist)
    
    
    
    peaks_image = np.zeros(aDicomFile.shape, np.bool)
    peaks_image[tuple(np.transpose(peeks))] = True
    
    seeds, numb_seed = ndi.label(peaks_image)
    ws = watershed(aDicomFile, seeds)
        
#     plt.subplot(2,2,3)
#     plt.imshow(mark_boundaries(aDicomFile, ws))  
    return ws, original_File  # return label reshaped label array of the clustered regions and the file itself  


def generate_dicom_dataset_watershed(source_folder):
    dicomSet = glob.glob(source_folder+"/**/*.dcm", recursive=True)
    dicomSet = sorted(dicomSet)
    
    shape = (160, 160)
    super_pixel_list = []
    
    for i, file in enumerate(dicomSet):
        theLabels, theDicomFile = get_watershed_superpixels(file)
        for l in np.unique(theLabels):
            aMask = np.zeros(shape)   #each mask task the shape of the image
            aMask[theLabels==l] = theDicomFile[theLabels==l]    
#             super_pixel_list.append([csr_matrix(aMask), i]) #compress the arrays as sparse column matrix and append to the list
            
            atuple = (csr_matrix(aMask), i)
            super_pixel_list.append(atuple) #compress the arrays as sparse column matrix and append to the list
    return super_pixel_list    



#test the segmentation
dicom_training_folder= "Training/DOI/Prostate3T-01-0002/"


