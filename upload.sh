#!/bin/bash

# ====== Config Section ======
GITHUB_USERNAME="iCherishxixixi"
REPO_NAME="MPTransformer"
GITHUB_TOKEN="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"  # replace with your real GitHub token
GIT_EMAIL="494778997@example.com"         # replace with your GitHub email

# ====== Git Initialization and Upload ======

# Set user identity
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GIT_EMAIL"

# Initialize local git repo
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit"

# Add remote origin with embedded token for authentication
git remote add origin https://$GITHUB_USERNAME:$GITHUB_TOKEN@github.com/$GITHUB_USERNAME/$REPO_NAME.git

# Set branch name to main
git branch -M main

# Push to GitHub
git push -u origin main
