#!/bin/bash
# Upload the canary-stamped eval logs (export/) to a Google Drive folder with rclone.
#
# One-time setup (interactive OAuth, run it yourself):  rclone config   -> new remote, type "drive", name "gdrive"
# Then:  bash scripts/upload_logs.sh [remote] [drive-folder-id]
set -e
REMOTE="${1:-gdrive}"
FOLDER_ID="${2:?usage: upload_logs.sh <rclone-remote> <drive-folder-id>}"

[ -d export/logs ] || { echo "export/logs missing — run: python scripts/export_logs.py"; exit 1; }

rclone copy export/ "$REMOTE:" --drive-root-folder-id "$FOLDER_ID" \
    --progress --transfers 8 --checkers 8 --drive-chunk-size 64M
echo "done"
