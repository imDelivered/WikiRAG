
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


import sys
import os
import time

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Enable Debug
from chatbot import config
config.DEBUG = True

from chatbot import chat
from chatbot.models import Message

TEST_QUERIES = [
    "Who assassinated Archduke Franz Ferdinand?",
    "Where was Abraham Lincoln shot?",
    "Who tried to assassinate Ronald Reagan in 1981?",
    "How did Julius Caesar die?",
    "Who killed Mahatma Gandhi?",
    "Who shot Pope John Paul II?",
    "What caused the Titanic to sink?",
    "Who was the first person to walk on the moon?",
    "When did the Berlin Wall fall?",
    "What date was the attack on Pearl Harbor?"
]

def run_test_query(query, index):
    print(f"\n{'='*80}")
    print(f"TEST #{index}: {query}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # 1. Build Messages (simulates Retrieval phase)
    system_prompt = "You are a helpful AI assistant. Answer the user's questions based on the provided context."
    history = [Message(role="user", content=query)]
    
    try:
        messages = chat.build_messages(system_prompt, history, user_query=query)
        
        # 2. Run Chat Generation
        model = config.DEFAULT_MODEL
        response = chat.full_chat(model, messages)
        
        elapsed = time.time() - start_time
        
        print(f"\n[RESULT #{index}]")
        print(f"Query: {query}")
        print(f"Time: {elapsed:.2f}s")
        print("-" * 40)
        print(response)
        print("-" * 40 + "\n")
        
    except Exception as e:
        print(f"ERROR on query '{query}': {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Starting Batch Test ({len(TEST_QUERIES)} queries)...")
    
    for i, query in enumerate(TEST_QUERIES, 1):
        run_test_query(query, i)
        # Small sleep to let logs clear/buffers flush
        time.sleep(1)
        
    print("\nBatch Test Complete.")