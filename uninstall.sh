#!/usr/bin/env bash
set -euo pipefail

echo "=== VaultRAG Uninstaller ==="

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 could not be found."
    exit 1
fi

# Check for tkinter (often separate package on Linux)
if ! python3 -c "import tkinter" &> /dev/null; then
    echo "Error: Python tkinter module not found."
    echo "Please install it with: sudo apt install python3-tk"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GUI_SCRIPT="$SCRIPT_DIR/uninstall_gui.py"

if [ ! -f "$GUI_SCRIPT" ]; then
    echo "Error: Could not find uninstall_gui.py"
    exit 1
fi

echo "Launching Uninstaller GUI..."
python3 "$GUI_SCRIPT"
echo "Done."
