#!/bin/bash

# === Config ===
GITHUB_USERNAME="iCherishxixixi"
REPO_NAME="MPTransformer"
GIT_EMAIL="494778997@example.com"  # Replace with your GitHub email

# === Navigate to target folder (optional, "." means current folder) ===
cd . || { echo "Folder not found"; exit 1; }

# Set Git identity
git config --global user.name "$GITHUB_USERNAME"
git config --global user.email "$GIT_EMAIL"

# Initialize Git if not already initialized
if [ ! -d ".git" ]; then
  git init
fi

# Remove existing remote if any
git remote remove origin 2>/dev/null

# Add SSH remote
git remote add origin git@github.com:$GITHUB_USERNAME/$REPO_NAME.git

# Add files and commit
git add .
git commit -m "Initial commit" || echo "Nothing new to commit"

# Rename and push to main branch
git branch -M main
git push -u origin main