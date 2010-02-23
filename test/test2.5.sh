#!/bin/sh -f

pyv="25"
d=`date +%Y-%m-%d`
outfile="test.$d.py$pyv"

/bin/rm $outfile
touch $outfile

for file in `cat testfiles`
do
echo $file >> $outfile
python2.5 $file 2>&1 >> $outfile
done

