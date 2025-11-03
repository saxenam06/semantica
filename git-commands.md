# Git Commands for Branch, PR, and Merge Workflow

## Option 1: Using GitHub CLI (Recommended)

### Step 1: Create and Switch to New Branch
```bash
git checkout -b feature/utils-implementation
```
or
```bash
git switch -c feature/utils-implementation
```

### Step 2: Stage Your Changes
```bash
git add libs/semantica/utils/
```

### Step 3: Commit Changes
```bash
git commit -m "feat: implement utils module with exceptions, helpers, logging, types, and validators

- Implement exceptions.py with comprehensive error handling
- Implement helpers.py with utility functions
- Implement logging.py with structured logging
- Implement types.py with type definitions and protocols
- Implement validators.py with validation functions
- Update __init__.py to export all utilities"
```

### Step 4: Push Branch to Remote
```bash
git push -u origin feature/utils-implementation
```

### Step 5: Create Pull Request (using GitHub CLI)
```bash
gh pr create --title "feat: implement utils module" --body "Implements complete utils module with all utilities, exceptions, logging, types, and validators" --base main
```

### Step 6: Merge Pull Request (using GitHub CLI)
```bash
gh pr merge feature/utils-implementation --merge --delete-branch
```

---

## Option 2: Manual Git Commands (Without GitHub CLI)

### Step 1: Create and Switch to New Branch
```bash
git checkout -b feature/utils-implementation
```

### Step 2: Stage Your Changes
```bash
git add libs/semantica/utils/
```

### Step 3: Commit Changes
```bash
git commit -m "feat: implement utils module with exceptions, helpers, logging, types, and validators

- Implement exceptions.py with comprehensive error handling
- Implement helpers.py with utility functions
- Implement logging.py with structured logging
- Implement types.py with type definitions and protocols
- Implement validators.py with validation functions
- Update __init__.py to export all utilities"
```

### Step 4: Push Branch to Remote
```bash
git push -u origin feature/utils-implementation
```

### Step 5: Create Pull Request
After pushing, go to GitHub and create PR, OR use the URL:
```
https://github.com/YOUR_USERNAME/semantica-9/compare/main...feature/utils-implementation
```

### Step 6: Merge Locally (Alternative to PR merge)
```bash
# Switch to main branch
git checkout main

# Pull latest changes
git pull origin main

# Merge feature branch
git merge feature/utils-implementation

# Push merged changes
git push origin main

# Delete local branch
git branch -d feature/utils-implementation

# Delete remote branch
git push origin --delete feature/utils-implementation
```

---

## Quick Commands Summary (Copy-Paste)

### For Feature Branch Workflow:
```bash
# 1. Create branch
git checkout -b feature/utils-implementation

# 2. Stage changes
git add libs/semantica/utils/

# 3. Commit
git commit -m "feat: implement utils module"

# 4. Push
git push -u origin feature/utils-implementation

# 5. Create PR (with GitHub CLI)
gh pr create --title "feat: implement utils module" --body "Implements complete utils module" --base main

# 6. Merge PR (with GitHub CLI)
gh pr merge feature/utils-implementation --merge --delete-branch
```

### For Direct Merge to Main (If PR not needed):
```bash
# 1. Create branch
git checkout -b feature/utils-implementation

# 2. Stage and commit
git add libs/semantica/utils/
git commit -m "feat: implement utils module"

# 3. Switch to main
git checkout main

# 4. Merge branch
git merge feature/utils-implementation

# 5. Push
git push origin main

# 6. Clean up
git branch -d feature/utils-implementation
```

---

## Notes:
- Replace `feature/utils-implementation` with your preferred branch name
- Replace commit message with your preferred message
- If using GitHub CLI, make sure you're logged in: `gh auth login`
- Always pull latest changes before creating branch: `git pull origin main`
