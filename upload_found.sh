#!/bin/bash
git config --global user.email "auto@scanner.com"
git config --global user.name "GPU Hunter Bot"
git pull
git add found.txt range.txt
git commit -m "Auto update results + range"
git push
