#!/bin/bash

# Create and push to github repo from the command line
# commit message is arg1

NOTE=$1
git add .
git commit -m $NOTE
git push origin master

