import numpy as np
from keras.utils import np_utils, generic_utils
from scipy.sparse import load_npz

'''
	Splits the train data into sub-arrays of Train, Val and Unlabeled Pool
'''
def split_train_X_pool(X_Train_all, Y_Train_all, img_rows, img_cols, nb_classes, X_Train_percent = .3):

    idx_negatives = np.array(np.where(Y_Train_all == 0)).reshape(-1) #search for all negative indices
    idx_positives = np.array(np.where(Y_Train_all == 1)).reshape(-1) #search for all positive indices

    train_num_half = int(X_Train_percent * idx_positives.shape[0]) # total number from training set to consider for initial training. 2x indicating positive and negative hence the name half

    X_Train_pos = X_Train_all[idx_positives[:train_num_half], :, :, :]  #select positive samples
    X_Train_neg = X_Train_all[idx_negatives[:train_num_half], :, :, :]  # select negative samples

    Y_Train_pos = Y_Train_all[idx_positives[:train_num_half]]  #concept similar to X_train_pos and X_train_neg
    Y_Train_neg = Y_Train_all[idx_negatives[:train_num_half]]

    X_Train = np.concatenate((X_Train_pos, X_Train_neg), axis=0)  #form the X train data
    Y_Train = np.concatenate((Y_Train_neg, Y_Train_pos), axis=0)  #form Y train data

    X_Pool_neg = X_Train_all[idx_negatives[train_num_half:], :, :, :]   #the remaining negative dataset is the negative pool
    Y_Pool_neg = Y_Train_all[idx_negatives[train_num_half:]]

    X_Pool_pos = X_Train_all[idx_positives[train_num_half:], :, :, :]   # the remaining positive dataset is the positive pool
    Y_Pool_pos = Y_Train_all[idx_positives[train_num_half:]]

    X_Pool = np.concatenate((X_Pool_neg, X_Pool_pos), axis=0)           # form x pool
    Y_Pool = np.concatenate((Y_Pool_neg, Y_Pool_pos), axis=0)           # form y pool

    #one -hot encode the vectors
    Y_Pool = np_utils.to_categorical(Y_Pool, nb_classes)
    Y_Train = np_utils.to_categorical(Y_Train, nb_classes)

    return X_Train, Y_Train, X_Pool, Y_Pool

def split_train(X_Train_all, Y_Train_all, img_rows, img_cols, nb_classes, X_Train_percent=.2, val_percent =.5):

    idx_negatives = np.array(np.where(Y_Train_all == 0)).reshape(-1) #search for all negative indices
    idx_positives = np.array(np.where(Y_Train_all == 1)).reshape(-1) #search for all positive indices

    # train_num_half = int(X_Train_percent * idx_positives.shape[0]) # total number from training set to consider for initial training. 2x indicating positive and negative hence the name half
    train_num_half  = 2
    X_Train_pos = X_Train_all[idx_positives[:train_num_half], :, :, :]  #select positive samples
    X_Train_neg = X_Train_all[idx_negatives[:train_num_half], :, :, :]  # select negative samples

    Y_Train_pos = Y_Train_all[idx_positives[:train_num_half]]  #concept similar to X_train_pos and X_train_neg
    Y_Train_neg = Y_Train_all[idx_negatives[:train_num_half]]

    X_Train = np.concatenate((X_Train_pos, X_Train_neg), axis=0)  #form the X train data
    Y_Train = np.concatenate((Y_Train_neg, Y_Train_pos), axis=0)  #form Y train data

    # print('X_Train and Y_Train shapes ', X_Train.shape, Y_Train.shape)
    # print('Distribution of Y_Train Classes:',
    # 	np.bincount(Y_Train.reshape(-1).astype(np.int)))

    left_over_after_xtrain_pos = idx_positives.shape[0] - train_num_half  #num remaining after picking positive samples for training set
    left_over_after_xtrain_neg = idx_negatives.shape[0] - train_num_half #num remaining after picking negative samples for training set

    val_pos_start_index = train_num_half  #calculate startng index for valication set
    val_pos_end_index = val_pos_start_index + int(
    	left_over_after_xtrain_pos * val_percent)      # calculate end index for validation set

    X_Valid_pos = X_Train_all[idx_positives[val_pos_start_index:
                                            val_pos_end_index], :, :, :]  #pick positive validation set
    Y_Valid_pos = Y_Train_all[idx_positives[val_pos_start_index:
                                            val_pos_end_index]]

    X_Valid_neg = X_Train_all[idx_negatives[val_pos_start_index:			#pick negative validation set
                                            val_pos_end_index], :, :, :]
    Y_Valid_neg = Y_Train_all[idx_negatives[val_pos_start_index:
                                            val_pos_end_index]]

    X_Valid = np.concatenate((X_Valid_pos, X_Valid_neg), axis=0)    # form validation set
    Y_Valid = np.concatenate((Y_Valid_pos, Y_Valid_neg), axis=0)
    # print('X_Valid and Y_Valid shapes ', X_Valid.shape, Y_Valid.shape)
    # print('Distribution of Y_Valid Classes:', np.bincount(Y_Valid.reshape(-1).astype(np.int)))

    X_Pool_neg = X_Train_all[idx_negatives[val_pos_end_index:], :, :, :]   #the remaining negative dataset is the negative pool
    Y_Pool_neg = Y_Train_all[idx_negatives[val_pos_end_index:]]

    X_Pool_pos = X_Train_all[idx_positives[val_pos_end_index:], :, :, :]   # the remaining positive dataset is the positive pool
    Y_Pool_pos = Y_Train_all[idx_positives[val_pos_end_index:]]

    X_Pool = np.concatenate((X_Pool_neg, X_Pool_pos), axis=0)			# form x pool
    Y_Pool = np.concatenate((Y_Pool_neg, Y_Pool_pos), axis=0)			# form y pool

    # print('X_Pool and Y_Pool shapes ', X_Pool.shape, Y_Pool.shape)
    # print('Distribution of Y_Pool Classes:',
    #       np.bincount(Y_Pool.reshape(-1).astype(np.int)))

    #one -hot encode the vectors
    Y_Valid = np_utils.to_categorical(Y_Valid, nb_classes)
    Y_Pool = np_utils.to_categorical(Y_Pool, nb_classes)
    Y_Train = np_utils.to_categorical(Y_Train, nb_classes)

    return X_Train, Y_Train, X_Valid, Y_Valid, X_Pool, Y_Pool

def split_train_ratio_based(X_Train_all, Y_Train_all, img_rows, img_cols, nb_classes, ratio =(60, 1), X_Train_percent=.2, val_percent =.5):

    idx_negatives = np.array(np.where(Y_Train_all == 0)).reshape(-1) #search for all negative indices
    idx_positives = np.array(np.where(Y_Train_all == 1)).reshape(-1) #search for all positive indices

    # train_num_half = int(X_Train_percent * idx_positives.shape[0]) # total number from training set to consider for initial training. 2x indicating positive and negative hence the name half
    train_num_half  = 2
    negatives_ratio = train_num_half * ratio[0]

    X_Train_pos = X_Train_all[idx_positives[:train_num_half], :, :, :]  #select positive samples
    X_Train_neg = X_Train_all[idx_negatives[:negatives_ratio], :, :, :]  # select negative samples

    Y_Train_pos = Y_Train_all[idx_positives[:train_num_half]]  #concept similar to X_train_pos and X_train_neg
    Y_Train_neg = Y_Train_all[idx_negatives[:negatives_ratio]]

    X_Train = np.concatenate((X_Train_pos, X_Train_neg), axis=0)  #form the X train data
    Y_Train = np.concatenate((Y_Train_neg, Y_Train_pos), axis=0)  #form Y train data

    # print('X_Train and Y_Train shapes ', X_Train.shape, Y_Train.shape)
    # print('Distribution of Y_Train Classes:',
    #   np.bincount(Y_Train.reshape(-1).astype(np.int)))

    left_over_after_xtrain_pos = idx_positives.shape[0] - train_num_half  #num remaining after picking positive samples for training set
    left_over_after_xtrain_neg = idx_negatives.shape[0] - negatives_ratio #num remaining after picking negative samples for training set

    val_pos_start_index = train_num_half  #calculate startng index for valication set
    val_pos_end_index = val_pos_start_index + int(
        left_over_after_xtrain_pos * val_percent)      # calculate end index for validation set

    X_Valid_pos = X_Train_all[idx_positives[val_pos_start_index:
                                            val_pos_end_index], :, :, :]  #pick positive validation set
    Y_Valid_pos = Y_Train_all[idx_positives[val_pos_start_index:
                                            val_pos_end_index]]

    X_Valid_neg = X_Train_all[idx_negatives[val_pos_start_index:            #pick negative validation set
                                            val_pos_end_index], :, :, :]
    Y_Valid_neg = Y_Train_all[idx_negatives[val_pos_start_index:
                                            val_pos_end_index]]

    X_Valid = np.concatenate((X_Valid_pos, X_Valid_neg), axis=0)    # form validation set
    Y_Valid = np.concatenate((Y_Valid_pos, Y_Valid_neg), axis=0)
    # print('X_Valid and Y_Valid shapes ', X_Valid.shape, Y_Valid.shape)
    # print('Distribution of Y_Valid Classes:', np.bincount(Y_Valid.reshape(-1).astype(np.int)))

    X_Pool_neg = X_Train_all[idx_negatives[val_pos_end_index:], :, :, :]   #the remaining negative dataset is the negative pool
    Y_Pool_neg = Y_Train_all[idx_negatives[val_pos_end_index:]]

    X_Pool_pos = X_Train_all[idx_positives[val_pos_end_index:], :, :, :]   # the remaining positive dataset is the positive pool
    Y_Pool_pos = Y_Train_all[idx_positives[val_pos_end_index:]]

    X_Pool = np.concatenate((X_Pool_neg, X_Pool_pos), axis=0)           # form x pool
    Y_Pool = np.concatenate((Y_Pool_neg, Y_Pool_pos), axis=0)           # form y pool

    # print('X_Pool and Y_Pool shapes ', X_Pool.shape, Y_Pool.shape)
    # print('Distribution of Y_Pool Classes:',
    #       np.bincount(Y_Pool.reshape(-1).astype(np.int)))

    #one -hot encode the vectors
    Y_Valid = np_utils.to_categorical(Y_Valid, nb_classes)
    Y_Pool = np_utils.to_categorical(Y_Pool, nb_classes)
    Y_Train = np_utils.to_categorical(Y_Train, nb_classes)

    return X_Train, Y_Train, X_Valid, Y_Valid, X_Pool, Y_Pool


def fetch_data(files, slice_range):

	 # randomly pick test_percent of folders
	num = len(files)
	XY_Data = list()
	#

	for f in files:
		Xy_tr = load_npz(f)
		Xy_tr= Xy_tr.toarray()
		trimed_data = None
		if slice_range != 0:
		    num_of_slices = Xy_tr[-1, 785]
		    start_slice = np.float(num_of_slices / 2) - np.floor(
		        slice_range / 2)
		    end_slice = start_slice + slice_range
		    start_indices = Xy_tr[:, 785] >= start_slice
		    end_indices = Xy_tr[:, 785] >= end_slice
		    intercept = start_indices & end_indices
		    trimed_data = Xy_tr[intercept]
		else:
		    trimed_data = Xy_tr
		if len(XY_Data) == 0:
		    XY_Data = trimed_data
		else:
			XY_Data = np.append(XY_Data, trimed_data, axis=0)
	return XY_Data
