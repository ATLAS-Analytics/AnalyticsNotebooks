#!/bin/bash
date
d=$(date +%H)
cd /home/ivukotic/workspace/AnalyticsNotebooks/ATLASalertsservice/
echo '----- general packet loss alerts -----'
/home/ivukotic/anaconda3/bin/jupyter nbconvert --execute Send\ packet\ loss\ alerts.ipynb --output outputs/res_Send_packet_loss_alerts_$d.ipynb --to notebook
