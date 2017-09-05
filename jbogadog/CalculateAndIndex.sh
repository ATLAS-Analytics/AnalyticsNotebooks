#!/bin/bash
echo "started at:"
date
startDate=$(date -u '+%Y-%m-%d' -d "-48hour")
echo "processing date" $startDate
cd /home/ivukotic/workspace/AnalyticsNotebooks/jbogadog/
export PATH="/home/ivukotic/anaconda3/bin:$PATH"
python Calculate\ and\ index\ rates\ and\ queue\ depths.py $startDate
