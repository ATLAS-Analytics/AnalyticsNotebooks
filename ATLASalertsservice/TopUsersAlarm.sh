#!/bin/bash
date
d=$(date +%d)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute TopUsersAlarm.ipynb  --output outputs/res_TopUsersAlarm_$d.ipynb --to notebook
