
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

"""Configuration constants."""

OLLAMA_CHAT_URL = "N/A" # Deprecated
# Local Model Repositories
MODEL_ALETHEIA_3B = "Ishaanlol/Aletheia-Llama-3.2-3B" 


DEFAULT_MODEL = MODEL_ALETHEIA_3B
STRICT_RAG_MODE = True
DEBUG = False

# Multi-Joint RAG System Configuration
USE_JOINTS = True

# Joint Models - All use the fast Aletheia 3B model
ENTITY_JOINT_MODEL = MODEL_ALETHEIA_3B
SCORER_JOINT_MODEL = MODEL_ALETHEIA_3B
FILTER_JOINT_MODEL = MODEL_ALETHEIA_3B
FACT_JOINT_MODEL = MODEL_ALETHEIA_3B

# Joint Temperatures
ENTITY_JOINT_TEMP = 0.1
SCORER_JOINT_TEMP = 0.0
FILTER_JOINT_TEMP = 0.1
FACT_JOINT_TEMP = 0.0

# Joint Timeout (not used for local inference but kept for compat)
JOINT_TIMEOUT = 10

# Adaptive RAG Configuration
ADAPTIVE_THRESHOLD = 4.0  # If max score is below this, trigger expansion

# Global Context Window Configuration
DEFAULT_CONTEXT_SIZE = 16384


