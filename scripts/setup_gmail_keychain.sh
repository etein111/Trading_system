#!/bin/zsh
set -euo pipefail
read "gmail_user?Gmail sender address: "
read -s "gmail_app_password?Gmail app password: "
echo
security add-generic-password -U -a "$gmail_user" -s trading-system-gmail-app-password -w "$gmail_app_password" >/dev/null
cat <<EOF
Saved the Gmail app password in macOS Keychain.
Set these variables in the automation environment:
  export TRADING_GMAIL_USER="$gmail_user"
  export GMAIL_TO="your-recipient@gmail.com"
EOF
