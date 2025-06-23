#!/bin/bash
set -e

# CONFIG
GITHUB_REPO="github.com/Soumya001/gpuhunt-realhash.git"
TOKEN="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"

# Git identity setup (do this once per machine if needed)
git config user.email "your@email.com"
git config user.name "Your Name"

# Sync & commit changes
git pull

# Stage and commit found.txt + range.txt
git add found.txt range.txt
git commit -m "Auto update results + range" || exit 0  # skip if no changes

# Push using token-based HTTPS URL
git push https://$TOKEN@$GITHUB_REPO
