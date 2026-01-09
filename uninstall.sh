#!/usr/bin/env bash

# Hermit - Offline AI Chatbot for Wikipedia & ZIM Files
# Copyright (C) 2026 Hermit-AI, Inc.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

set -euo pipefail

echo "=== Hermit Uninstaller ==="

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