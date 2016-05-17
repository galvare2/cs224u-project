#!/bin/bash

# Create and push to github repo from the command line
# commit message is arg1

note=$1
git add .
git commit -m note
git push origin master

