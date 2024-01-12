#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 (3 | 4)"
    exit 1
fi

dims=$1
re='^[34]$'
if ! [[ $dims =~ $re ]] ; then
    echo "Usage: $0 (3 | 4)"
    exit 2
fi

if [ $dims -eq 3 ]; then
    runpath="runs_3d"
else
    runpath="runs_4d"
fi
echo "EAM Hetero experiments."
echo "Storing results in $runpath"
echo "=================== Starting..."
date
python eam.py -n mnist --dims=$dims --runpath=$runpath && \
python eam.py -n fashion --dims=$dims --runpath=$runpath && \
python eam.py -f mnist --dims=$dims --runpath=$runpath && \
python eam.py -f fashion --dims=$dims --runpath=$runpath && \
python eam.py -s mnist --dims=$dims --runpath=$runpath && \
python eam.py -s fashion --dims=$dims --runpath=$runpath && \
python eam.py -e --dims=$dims --runpath=$runpath && \
python eam.py -v --dims=$dims --runpath=$runpath && \
python eam.py -w --dims=$dims --runpath=$runpath && \
python eam.py -r --dims=$dims --runpath=$runpath
echo -n "=================== "
ok=$?
date
if [ $ok -eq 0 ]; then
    echo "Done."
else
    echo "Sorry, something went wrong."
fi

