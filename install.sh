#!/usr/bin/env sh
set -eu

if command -v uvx >/dev/null 2>&1; then
  exec uvx --from . collama "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m colla "$@"
fi

echo "python3 is required to run collama-on-a-leash." >&2
exit 1
