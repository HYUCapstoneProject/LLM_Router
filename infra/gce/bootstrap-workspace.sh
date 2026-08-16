#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  gh \
  jq \
  tmux \
  unzip \
  build-essential \
  docker.io \
  docker-compose-v2

curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh
sudo -E bash /tmp/nodesource_setup.sh
sudo apt-get install -y nodejs
rm -f /tmp/nodesource_setup.sh

sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

sudo npm install --global @openai/codex

mkdir -p "$HOME/work"
if [[ ! -d "$HOME/work/LLM_Router/.git" ]]; then
  if gh auth status >/dev/null 2>&1; then
    gh repo clone HYUCapstoneProject/LLM_Router "$HOME/work/LLM_Router"
  else
    echo "GitHub authentication is required before cloning the private repository."
  fi
fi

git config --global init.defaultBranch main
git config --global pull.ff only

cat <<'EOF'

Workspace bootstrap complete.

Next interactive steps:
  1. Reconnect SSH so the docker group membership takes effect.
  2. Run: codex
  3. Run: gh auth login --git-protocol ssh --web
  4. Run: gh repo clone HYUCapstoneProject/LLM_Router ~/work/LLM_Router
  5. Work in: ~/work/LLM_Router
EOF
