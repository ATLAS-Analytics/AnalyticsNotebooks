#!/bin/bash
date
d=$(date +%H_%M)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute CheckClusterState.ipynb  --output outputs/res_CheckClusterState_$d.ipynb --to notebook
