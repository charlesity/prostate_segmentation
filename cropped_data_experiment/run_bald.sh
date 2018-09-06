#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"

source ~/charles/environments/keras0/bin/activate

echo "Running script without oversampling"
python ~/charles/prostate_segmentation/cropped_data_experiment/active_deep_seg_bald.py -ds_type 0

echo "Running script with oversampling"
python ~/charles/prostate_segmentation/cropped_data_experiment/active_deep_seg_bald.py -ds_type 0 -ovs
