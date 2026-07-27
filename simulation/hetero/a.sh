#!/bin/bash



num_trimers=(10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 200 250 300 400 500 600 700 800 900)
num_rnas=(3 5 10 15 20 25 30 35 40 50 60 70 80 90 100 120 130 140 150 160)
Ly=(80)
kd=(0 5 100)

for((i=2;i<=2;i++)) #loop for config with same parameters
do
	for((j=0;j<=5;j++))
	do
		for((k=0;k<=19;k++))
		do
			for((l=0;l<=0;l++))
			do
				for((m=0;m<=2;m++))
				do
					python simulation3.py $i ${num_trimers[${j}]} ${num_rnas[${k}]} ${Ly[${l}]} ${kd[${m}]}
				done
			done
		done
	done
done 
 
	