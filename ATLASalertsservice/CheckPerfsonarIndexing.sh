#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute CheckPerfsonarIndexing.ipynb --output outputs/res_CheckPerfsonarIndexing_$d.ipynb --to notebook
