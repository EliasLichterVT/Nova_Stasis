#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
bootstrap="${NOVA_LAUNCHER_HOME:-$HOME/.nova-stasis}/bootstrap"
uv="$bootstrap/uv"
mkdir -p "$bootstrap"
if [[ ! -x "$uv" ]]; then
    installer="$(mktemp)"
    trap 'rm -f "$installer"' EXIT
    curl --fail --location --retry 3 --output "$installer" https://astral.sh/uv/0.9.30/install.sh
    UV_UNMANAGED_INSTALL="$bootstrap" sh "$installer"
    rm -f "$installer"
    trap - EXIT
fi
"$uv" run --no-project --no-config --managed-python --python 3.12 -- python "$root/launcher.py" "$@"
