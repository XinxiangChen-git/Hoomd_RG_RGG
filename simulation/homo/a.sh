#!/bin/bash


n_polymer=(500 600 700 800 900 1000 1200 1500 1700 2000 2200 2500 2700 3000 3200 3500 3700 4000)
N_length=(65)
f_sticker=(10)
L_final=(80)
e_a=(12.5 13.5 7 8 9 6 10 11 12 13 3 4 5 6 7 8 10 12 14 16 18 20 22 24 26)

for((i=1;i<=1;i++)) #loop for config with same parameters
do	   
	for((j=12;j<=13;j++))
	do
		for((k=0;k<=0;k++))
		do
			for((l=0;l<=0;l++))
			do
				for((m=0;m<=0;m++))
				do
					for((n=0;n<=0;n++))
					do
						python teset_v4_1.py $i ${n_polymer[${j}]} ${N_length[${k}]} ${f_sticker[${l}]} ${L_final[${m}]} ${e_a[${n}]}
					done
				done
			done
		done
	done
done

