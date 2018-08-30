#~/bin/bash

echo "Welcome"
sleep 1
echo "Starting script"

python -W ignore active_deep_seg_all.py

git add Results/
git commit -m "updated results"
git push origin master