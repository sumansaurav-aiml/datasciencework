#!/bin/bash
export PYTHONPATH="$PWD"

echo $PYTHONPATH

python3 ./src/main.py

if [ $? -eq 0 ]
then
    echo "Successfully executed job"
    echo "copying files"
    bucket_name="aiml-recommendersystem-data"
    date=$(date '+%Y-%m-%d')
    cd src/data
    aws s3 sync . s3://${bucket_name}/${ENV}
    #/${date} --exclude "jobid*/*.log"
else
    echo "Job exited with error." >&2
    echo "copying files"
    bucket_name="aiml-recommendersystem-data"
    date=$(date '+%Y-%m-%d')
    cd src/data
    aws s3 sync . s3://${bucket_name}/${ENV}    
    exit 1
fi

