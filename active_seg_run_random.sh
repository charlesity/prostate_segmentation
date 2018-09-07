#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"


source ~/charles/environments/keras0/bin/activate

echo "Running script on 60 by 60 zero based data"
python ~/charles/prostate_segmentation/active_deep_seg_random.py
