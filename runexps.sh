#!/bin/bash

left_ds=mnist
right_ds=emnist
runpath=runs

echo "EAM Hetero experiments."
echo "Storing results in $runpath"
echo "=================== Starting at `date`"
python eam.py -n $left_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -n $right_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -f $left_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -f $right_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -c $left_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -c $right_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -s $left_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -s $right_ds --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -e --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -v --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -w --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -q --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -P --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -p --runpath=$runpath && \
echo "------------------- `date`" && \
python eam.py -r --runpath=$runpath && \
echo "=================== Ending at `date`"
ok=$?
if [ $ok -eq 0 ]; then
    echo "Done."
else
    echo "Sorry, something went wrong."
fi

