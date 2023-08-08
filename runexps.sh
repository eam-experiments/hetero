#!/bin/bash

runpath="runs"
ok=1
echo "EAM Hetero experiments."
echo "Storing results in $runpath"
echo "=================== Starting..."
python eam.py -n mnist --runpath=$runpath && \
python eam.py -n fashion --runpath=$runpath && \
python eam.py -f mnist --runpath=$runpath && \
python eam.py -f fashion --runpath=$runpath && \
python eam.py -s mnist --runpath=$runpath && \
python eam.py -s fashion --runpath=$runpath && \
python eam.py -e --runpath=$runpath && \
python eam.py -r --runpath=$runpath
echo "=================== Done."

