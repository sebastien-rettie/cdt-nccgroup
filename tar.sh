#!/bin/sh

#UNTARS ALL FILES WITH .tar FILETYPE IN LOCAL DIR

for file in ./*.tar
do
	echo ""
	echo "UNZIPPING $file ..."
	sleep 0.5s

	tar -xvf "$file"

	sleep 1s
done
