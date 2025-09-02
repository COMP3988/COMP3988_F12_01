# Github/Git Standard Operating Procedure

## Basic Commands
```bash
# Clone repo to local directory
git clone git@github.sydney.edu.au:COMP3988/COMP3988_F12_01.git

# Stage all items in current directory
git add .

# Take all staged changes and create a commit with an informative message, following convention
git commit -m "feat: added login page"

# Push all new commits to the current branch
git push

# Checkout to and create a new branch, following naming convention
git checkout -b bugfix/duplicate_records

# Checkout to existing branch
git checkout bugfix/duplicate_records

# Pull changes to current branch from origin
git pull

# Merge with specified branch
git merge main

# Rebase with specified branch
git rebase main
```


## Branch Naming Conventions
- **feature/** for new features (e.g., `feature/user-auth`)
- **bugfix/** for bug fixes (e.g., `bugfix/login-error`)
- **hotfix/** for urgent fixes (e.g., `hotfix/critical-crash`)
- **docs/** for documentation changes (e.g., `docs/update-readme`)

## Commit Message Guidelines
- Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):
	- `feat:` for new features
	- `fix:` for bug fixes
	- `docs:` for documentation
	- `refactor:` for code changes that neither fix a bug nor add a feature
	- `test:` for adding or updating tests
- Example: `feat: implement user login with JWT`

## Pull Requests (PRs)
- Always create a PR for merging into `main` or `develop`.
- Assign reviewers and add a clear description of changes.
- Link related issues (e.g., `Closes #12`).
- Ensure all checks pass before requesting review.

## Resolving Merge Conflicts
1. Pull the latest changes: `git pull origin main`
2. If conflicts occur, open the conflicted files and resolve them manually.
3. After resolving, add the files: `git add <file>`
4. Continue merge or rebase:
	 - For merge: `git commit`
	 - For rebase: `git rebase --continue`
5. Push your changes.

## Stashing Changes
Temporarily save uncommitted changes:
```bash
git stash
```
Apply stashed changes later:
```bash
git stash pop
```

## Undoing Changes
- Discard changes in a file: `git checkout -- <file>`
- Unstage a file: `git reset HEAD <file>`
- Amend last commit: `git commit --amend`
- Revert a commit: `git revert <commit-hash>`

## Best Practices
- Pull frequently to stay up to date.
- Write clear, concise commit messages.
- Keep branches focused and small.
- Delete branches after merging.
- Never commit sensitive information.

## Troubleshooting
- **Authentication issues:** Ensure your SSH keys are added to your GitHub account.
- **Detached HEAD:** Checkout a branch before making changes: `git checkout <branch>`
- **Accidentally committed to wrong branch:**
	1. Create a new branch: `git checkout -b correct-branch`
	2. Reset the original branch if needed.

---
For more, see the [Pro Git Book](https://git-scm.com/book/en/v2) or your team's Git workflow documentation.

