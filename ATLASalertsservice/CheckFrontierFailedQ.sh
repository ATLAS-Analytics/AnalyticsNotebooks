#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute CheckFrontierNotServed.ipynb --output outputs/res_CheckFrontierNotServed_$d.ipynb --to notebook
