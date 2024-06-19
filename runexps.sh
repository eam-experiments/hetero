#!/bin/bash

left_ds=mnist
right_ds=emnist
runpath=runs

echo "EAM Hetero experiments."
echo "Storing results in $runpath"
echo "=================== Starting at `date`"
python eam.py -n $left_ds --runpath=$runpath && \
python eam.py -n $right_ds --runpath=$runpath && \
python eam.py -f $left_ds --runpath=$runpath && \
python eam.py -f $right_ds --runpath=$runpath && \
python eam.py -c $left_ds --runpath=$runpath && \
python eam.py -c $right_ds --runpath=$runpath && \
python eam.py -s $left_ds --runpath=$runpath && \
python eam.py -s $right_ds --runpath=$runpath && \
python eam.py -e --runpath=$runpath && \
python eam.py -v --runpath=$runpath && \
python eam.py -w --runpath=$runpath && \
python eam.py -q --runpath=$runpath && \
python eam.py -r --runpath=$runpath && \
python eam.py -P constructed --runpath=$runpath && \
python eam.py -P recalled --runpath=$runpath && \
python eam.py -P extracted --runpath=$runpath && \
python eam.py -p constructed --runpath=$runpath && \
python eam.py -p recalled --runpath=$runpath && \
python eam.py -p extracted --runpath=$runpath && \
python eam.py -u --runpath=$runpath && \
echo "=================== Ending at `date`"
ok=$?
if [ $ok -eq 0 ]; then
    echo "Done."
else
    echo "Sorry, something went wrong."
fi

