#!/bin/bash

# Exit if anything fails
set -e

# GitHub repo details
REPO="Soumya001/gpuhunt-realhash"
BRANCH="main"
TOKEN="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"

# Set Git config (optional but recommended)
git config user.name "autopush-bot"
git config user.email "autopush@example.com"

# Stage the key files
git add found.txt range.txt || true

# Commit only if there are changes
if git diff --cached --quiet; then
  echo "[*] No new changes to commit."
else
  git commit -m "Auto update results + range"
  git push "https://${TOKEN}@github.com/${REPO}.git" "$BRANCH"
fi
