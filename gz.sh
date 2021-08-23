#!/bin/sh

#UNZIPS ALL FILES WITH .gz FILETYPE IN LOCAL DIR

for file in ./*.gz
do
	echo ""
	echo "UNZIPPING $file ..."
	sleep 0.5s

	gzip -d "$file"

	sleep 1s
done
