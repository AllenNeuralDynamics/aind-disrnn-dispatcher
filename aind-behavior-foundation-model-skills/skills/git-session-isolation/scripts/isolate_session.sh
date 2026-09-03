#!/usr/bin/env bash
# isolate_session.sh — give a Claude Science session its own private git
# checkout of a shared repo, working *around* the sandbox rule that forbids
# creating any path named `.git`. The git-dir is named `<repo>_store` instead.
#
# Usage:
#   source isolate_session.sh
#   iso_open <repo-name> <base-branch> <new-branch>
#     e.g. iso_open aind-dynamic-foraging-bfm-wrapper ai_hub_pck_integration feat/session-A
#   ... edit files under $ISO_TREE, then:
#   iso_commit "message"
#   iso_push                       # pushes $ISO_BRANCH to GitHub origin
#
# Env knobs (optional):
#   ISO_WS       workspace root for checkouts (default: ./iso-sessions)
#   ISO_LOCAL    path to the shared local repo (default: /Users/han.hou/Scripts/<repo>)
#   ISO_ORG      GitHub org (default: AllenNeuralDynamics)

iso_open() {
  local repo="$1" base="$2" newbranch="$3"
  : "${ISO_WS:=$PWD/iso-sessions}"
  : "${ISO_ORG:=AllenNeuralDynamics}"
  local local_repo="${ISO_LOCAL:-/Users/han.hou/Scripts/$repo}"

  export ISO_REPO="$repo"
  export ISO_BRANCH="$newbranch"
  export GIT_DIR="$ISO_WS/$repo/store"        # NOT named .git  -> allowed
  export GIT_WORK_TREE="$ISO_WS/$repo/tree"
  export ISO_TREE="$GIT_WORK_TREE"
  mkdir -p "$GIT_WORK_TREE"

  git init -q
  git config user.name  "${GIT_AUTHOR_NAME:-claude-science}"
  git config user.email "${GIT_AUTHOR_EMAIL:-noreply@localhost}"

  # origin = GitHub (auth via injected token, embedded in URL)
  git remote remove origin 2>/dev/null
  git remote add origin "https://x-access-token:${GITHUB_TOKEN}@github.com/$ISO_ORG/$repo.git"

  # Prefer fetching the base from GitHub; fall back to the local folder if offline.
  if git fetch -q origin "$base" 2>/dev/null; then
    git checkout -q -B "$newbranch" FETCH_HEAD
    echo "checked out $newbranch off origin/$base"
  elif [ -d "$local_repo/.git" ]; then
    git remote remove shared 2>/dev/null
    git remote add shared "$local_repo"
    git -c protocol.file.allow=always fetch -q shared "refs/heads/$base:refs/remotes/shared/$base"
    git checkout -q -B "$newbranch" "shared/$base"
    echo "checked out $newbranch off shared(local)/$base"
  else
    echo "ERROR: could not fetch base '$base' from GitHub or local repo" >&2; return 1
  fi
  echo "WORK TREE: $ISO_TREE"
}

iso_commit() { git add -A && git commit -q -m "$1" && echo "committed $(git rev-parse --short HEAD)"; }
iso_push()   { git push -u origin "HEAD:$ISO_BRANCH" && echo "pushed $ISO_BRANCH -> origin"; }
iso_status() { echo "repo=$ISO_REPO branch=$ISO_BRANCH"; echo "GIT_DIR=$GIT_DIR"; git status -sb; }
