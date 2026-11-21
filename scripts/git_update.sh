#!/bin/bash

# Stop at the first error
set -e

echo "=== Git Update ==="

# Stage all changes that are not git-ignored
git add -A

# Commit
git commit -m "Update"

# Push to GitHub
git push

echo "=== Push completed! ==="