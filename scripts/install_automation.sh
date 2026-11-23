#!/bin/zsh
set -euo pipefail
project_root="$(cd "$(dirname "$0")/.." && pwd)"
read "gmail_user?Gmail sender address: "
read "gmail_to?Recipient Gmail address: "
read -s "gmail_app_password?Gmail app password: "
echo
security add-generic-password -U -a "$gmail_user" -s trading-system-gmail-app-password -w "$gmail_app_password" >/dev/null
cat > "$project_root/automation.env" <<EOF
TRADING_GMAIL_USER=$gmail_user
GMAIL_TO=$gmail_to
EOF
chmod 600 "$project_root/automation.env"
mkdir -p "$project_root/research/daily_reports"
mkdir -p "$HOME/Library/LaunchAgents"
for session in pre intra post; do
  case "$session" in
    pre) hour=22; minute=20; flags="--refresh";;
    intra) hour=1; minute=0; flags="--refresh";;
    post) hour=5; minute=10; flags="--refresh --retrain --execute-paper";;
  esac
  label="com.chuan.trading-system.$session"
  plist="$HOME/Library/LaunchAgents/$label.plist"
  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key><array>
    <string>/bin/zsh</string><string>-lc</string>
    <string>source "$project_root/automation.env"; exec "$project_root/.venv/bin/python" "$project_root/scripts/run_daily_report.py" --session $session --config "$project_root/config_qlib.yaml" --data "$project_root/research/expanded_data/prices.csv" --state "$project_root/paper/account.json" --log "$project_root/paper/trades.csv" --scores "$project_root/research/qlib_model_enhanced/qlib_scores.csv" --candidates-file "$project_root/research/expanded_data/symbols.txt" --output-dir "$project_root/research/daily_reports" --refresh $flags</string>
  </array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>$hour</integer><key>Minute</key><integer>$minute</integer></dict>
  <key>StandardOutPath</key><string>$project_root/research/daily_reports/$session.launchd.log</string>
  <key>StandardErrorPath</key><string>$project_root/research/daily_reports/$session.launchd.error.log</string>
</dict></plist>
EOF
  launchctl bootout "gui/$(id -u)/$label" 2>/dev/null || true
  launchctl bootstrap "gui/$(id -u)" "$plist"
done
cat <<EOF
Installed pre 22:20, intra 01:00, post 05:10 in Hong Kong local time.
Reports: $project_root/research/daily_reports
Paper account updates only in the post run; all orders are whole-share quantities.
EOF
