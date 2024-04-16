#!/bin/bash

echo "===== This will start upload the feature batch files to Neuronpedia."
echo "===== You'll need Neuronpedia running at localhost:3000 for this to work."

echo ""
echo "(Step 1 of 1)"
echo "What is the absolute, full local DIRECTORY PATH to your Neuronpedia batch outputs?"
read outputfilesdir

echo ""
read -p "===== Hit ENTER to start uploading!" start

echo "RUNNING: python upload_batch.py $outputfilesdir"
python upload_batch.py $outputfilesdir

echo ""
echo "===== ALL DONE."
echo "===== Go to http://localhost:3000 to browse your features"