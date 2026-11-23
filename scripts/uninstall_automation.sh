#!/bin/zsh
set -euo pipefail
for session in pre intra post; do
  label="com.chuan.trading-system.$session"
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  rm -f "$HOME/Library/LaunchAgents/$label.plist"
done
echo "Trading system automations removed."
