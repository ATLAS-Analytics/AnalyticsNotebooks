#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute CheckJobAndTaskIndexing.ipynb --output outputs/res_CheckJobAndTaskIndexing_$d.ipynb --to notebook  
