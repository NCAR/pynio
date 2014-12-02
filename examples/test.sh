#!/bin/sh -f

pyv="27"
d=`date +%Y-%m-%d`
outfile="test.$d.py$pyv"

/bin/rm $outfile
touch $outfile

for file in `cat testfiles`
do
echo $file 
python2.7 $file 2>&1 
done | tee $outfile

