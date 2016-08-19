#!/bin/bash
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/home/ivukotic/anaconda3/bin/jupyter nbconvert --execute CheckPerfsonarIndexing.ipynb --output outputs/res_CheckPerfsonarIndexing_$d.ipynb --to notebook
