#!/bin/bash
date
d=$(date +%d)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/home/ivukotic/anaconda3/bin/jupyter nbconvert --execute TopUsersAlarm.ipynb  --output outputs/res_TopUsersAlarm_$d.ipynb --to notebook
