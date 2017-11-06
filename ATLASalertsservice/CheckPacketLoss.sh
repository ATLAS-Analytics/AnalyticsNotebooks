#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
/usr/bin/jupyter nbconvert --execute CheckPacketLoss.ipynb --output outputs/res_CheckPacketLoss_$d.ipynb --to notebook
