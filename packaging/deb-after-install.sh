#!/bin/sh
set -e

SANDBOX_PATH="/opt/Overmind/chrome-sandbox"

if [ -f "$SANDBOX_PATH" ]; then
  chown root:root "$SANDBOX_PATH"
  chmod 4755 "$SANDBOX_PATH"
fi
