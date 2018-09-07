#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"


source ~/charles/environments/keras0/bin/activate
python -W ignore active_deep_seg_all.py
