#!/bin/bash

find . | grep "__pycache__$" | xargs rm -r

git add .
git commit -am $1
git push


