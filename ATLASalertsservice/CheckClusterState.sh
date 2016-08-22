#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/home/ivukotic/anaconda3/bin/jupyter nbconvert --execute CheckClusterState.ipynb  --output outputs/res_CheckClusterState_$d.ipynb --to notebook
