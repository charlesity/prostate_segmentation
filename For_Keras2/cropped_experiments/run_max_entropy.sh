#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"

# source ~/charles/environments/keras0/bin/activate

# echo "Running script without oversampling"
# python ./active_deep_seg_max_entropy.py -ds_type 0 -nexp 1

echo "Running script with oversampling"
python ./active_deep_seg_max_entropy.py -ds_type 0 -ovs -nexp 3
