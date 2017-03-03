#!/bin/zsh
for ((i = 22; i < 23; i++)); 
do 
 ./CheckPacketLoss.py 2017 1 $i 0
 ./CheckPacketLoss.py 2017 1 $i 3
 ./CheckPacketLoss.py 2017 1 $i 6
 ./CheckPacketLoss.py 2017 1 $i 9
 ./CheckPacketLoss.py 2017 1 $i 12
 ./CheckPacketLoss.py 2017 1 $i 15
 ./CheckPacketLoss.py 2017 1 $i 18
 ./CheckPacketLoss.py 2017 1 $i 21 
done
