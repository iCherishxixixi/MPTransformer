#!/bin/bash

# === Config ===
GITHUB_USERNAME="iCherishxixixi"
REPO_NAME="MPTransformer"
GIT_EMAIL="494778997@qq.com"
TARGET_BRANCH="init-upload"

# === Navigate to current folder ===
cd . || { echo "Folder not found"; exit 1; }

# Set Git identity
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GIT_EMAIL"

# Initialize Git repo if needed
if [ ! -d ".git" ]; then
  git init
fi

# Remove old origin (if any)
git remote remove origin 2>/dev/null

# Add SSH remote
git remote add origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git

# Create and switch to new feature branch
git checkout -b $TARGET_BRANCH

# Add files and commit
git add .
git commit -m "Initial commit via pull request" || echo "Nothing new to commit"

# Push to new branch
git push -u origin $TARGET_BRANCH

# Reminder
echo "Push complete. Open the following link to create a Pull Request:"
echo "https://github.com/$GITHUB_USERNAME/$REPO_NAME/compare"
