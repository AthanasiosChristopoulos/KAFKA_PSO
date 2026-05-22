

## ===========================================================================
## Git: ======================================================================

```bash

# ===================================================================
# Github being unbelievably annoying

git config --global user.name "AthanasiosChristopoulos"
git config --global user.email "athanasioschristopoulos61@gmail.com"

# Codes:
	username: AthanasiosChristopoulos
	password: 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

# ===================================================================
# Git general

git clone https://github.com/AthanasiosChristopoulos/Kafka_PSO.git
git clone -b DL4J-PSO-Server-2 --single-branch https://github.com/AthanasiosChristopoulos/KAFKA_PSO_4.git
git push https://github.com/AthanasiosChristopoulos/Kafka_PSO.git

git branch
git branch -d branch_name   # Delete a branch
git branch -D branch_name

git log --oneline

git branch -m DL4J-gBest

# Creating Repository:
git init
git remote add origin https://github.com/AthanasiosChristopoulos/WifiDoctor.git

# git ===============================================================================

git add .
git add Documentantion.txt

git commit -m "Your commit message" (--amend)
git commit --amend --no-edit  # Amend (edit last commit, dont create new one) the commit without changing the commit message

# Manage commits:
git log 					# View Commits
git reset --soft HEAD~1   	# Removes commit but keeps changes
git reset --hard HEAD~1   	# Removes commit AND changes

# ==================================================================
# Remote - Githup repo ===================================================================

git remote -v
git remote set-url origin https://github.com/AthanasiosChristopoulos/map-explorer-frontend.git
git remote add origin https://github.com/AthanasiosChristopoulos/map-explorer-frontend.git
git remote add origin https://github.com/AthanasiosChristopoulos/cv.git

git remote remove origin
git remote add origin <new-repo-url>

git push -u origin main

git fetch --prune origin    # fetch does NOT modify your code or merge anything.
                            # It only updates Git’s knowledge of the remote.

git branch -r       # This command shows remote-tracking branches that your local repo currently knows about.

git branch -vv      # Shows branches on local and what they track on remote     


# Restricted fetch configuration in .git/config:
git config --get remote.origin.fetch    # Which branches are allowed to be tracked
git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"    # Set to be able to fetch everything

git reset --soft HEAD~1		# Roll back last commit

# ==================================================================
# git resolve merge conflicts: =====================================

# Local be dominant:
git push --force origin main 	# (so you dont have to pull first / be up to date)
git push --force origin DL4J-PSO-federated
# Merge and Resolve:
git pull --no-rebase	# do a merge, dont rebase, if there are overlapping edits, Git will stop and ask you to resolve them manually
						# Re-applies your commits on top of the latest remote commits
						
# Remote be dominant:
git fetch origin
git reset --hard origin/<Branch name>
git reset --hard origin/DL4J-PSO-federated
git clean -fd

# ===================================================================
# .gitignore - Remove something you already tracked:

# you will need this to check what is already staged:
git ls-files	# for files
git ls-files | xargs -n1 dirname | sort -u 	# for directories

# for files:
git restore --staged main.bbl main.blg main.fls main.out main.tex
git rm --cached main.bbl main.blg main.fls main.out main.tex

git restore --staged presentation.pdf
git rm --cached presentation.pdf

git rm --cached generated/mylexer

# for directories:
git reset
git rm -r --cached node_modules
git rm -r --cached .vscode

```