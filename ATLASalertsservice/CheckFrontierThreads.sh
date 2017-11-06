#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute CheckFrontierThreads.ipynb --output outputs/res_CheckFrontierThreads_$d.ipynb --to notebook
