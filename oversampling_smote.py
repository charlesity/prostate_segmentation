from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import scipy as sp
import random
random.seed(0)
import scipy.io
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, KernelPCA

from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from mpl_toolkits.mplot3d import Axes3D



img_dim = (28**2)
XY_Data = np.load("./Processed_data/XY_Dataset_28_28.npy")

X = XY_Data[:, :img_dim]
y = XY_Data[:, img_dim]

slice_number_feature = True

if slice_number_feature:
    f_slice = np.array([1 if a/18 > 1 else a/18 for a in XY_Data[:, 785]]) 
    X = np.c_[X, f_slice]

random_set = random.sample(range(0, X.shape[0]), X.shape[0]) 

subset = 10000
X = X[random_set[:subset]]
y = y[random_set[:subset]] 

def plot_resampling(ax, X, y, title):
    c0 = ax.scatter(X[y == 0, 0], X[y == 0, 1], label="Class #0", alpha=0.5)
    c1 = ax.scatter(X[y == 1, 0], X[y == 1, 1], label="Class #1", alpha=0.5)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([-6, 8])
    ax.set_ylim([-6, 6])

    return c0, c1


ada = ADASYN(random_state=0)
sm = SMOTE(random_state=0)
rand = RandomOverSampler(random_state=0)
pca = KernelPCA(n_components=3, kernel="rbf", fit_inverse_transform=True, gamma=10)

methods = [sm, ada, rand]
X_resampled = []
y_resampled = []
X_res_vis = []


for index, method in enumerate(methods):
    print ("Over sampling using method "+ str(index))
    X_res, y_res = method.fit_sample(X, y)
    X_resampled.append(X_res)
    y_resampled.append(y_res)
    X_res_vis.append(pca.fit_transform(X_res))

X_res_vis.append(X)
y_resampled.append(y)

for i in range(len(methods)+1):    
    print ("Visualizing the "+ str(i)+"th method out of "+str(len(methods))+" methods")
    # # Two subplots, unpack the axes array immediately
    fig = plt.figure()
    # f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = fig.add_subplot(32, projection='3d')

    ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[y == 0, 0], X[y == 0, 1],X[y == 0, 2],  label="Class #0", alpha=0.5)
    ax.scatter(X_res_vis [i][y_resampled[i] == 0, 0], X_res_vis [i][y_resampled[i] == 0, 1], X_res_vis [i][y_resampled[i] == 0, 2],  label="Class #0", alpha=0.5)
    ax.scatter(X_res_vis [i][y_resampled[i] == 1, 0], X_res_vis [i][y_resampled[i] == 1, 1], X_res_vis [i][y_resampled[i] == 1, 2],  label="Class #1", alpha=0.5)
    
    y_ps = (y_resampled[i][y_resampled[i] == 1]).shape[0]
    y_neg = (y_resampled[i][y_resampled[i] == 0]).shape[0]

    if i == 0:
        ax.set_title("Over Samplying Via SMOTE and Slice Number Feature {} No: Pos = {} No: Neg  ={}".format(slice_number_feature, y_ps, y_neg))        
    elif i == 1:
        ax.set_title("Over Samplying Via ADASYN and Slice NO. Feature {}  No: Pos = {} No: Neg  ={}".format (slice_number_feature, y_ps, y_neg))
    elif i == 2:
        ax.set_title("Random Over Samplying and Slice Number Feature {} No: Pos = {} No: Neg  ={} ".format (slice_number_feature, y_ps, y_neg))
    else:
        ax.set_title("No Over Samplying and Slice Number Feature {} No: Pos = {} No: Neg  = {} ".format(slice_number_feature, y_ps, y_neg))

    ax.set_xlabel('First Component')
    ax.set_ylabel('Second Component')
    ax.set_zlabel('Third Component')
    plt.legend()
plt.show()
