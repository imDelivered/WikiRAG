#!/usr/bin/env python3
"""
Main entry point for Kiwix RAG application.
This is a refactored version that maintains backward compatibility.
"""

# Import everything from the original file for backward compatibility
# This allows the refactored code to work while we transition
import sys
import os

# Add parent directory to path so we can import from kiwix_chat module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For now, import from the original file to maintain functionality
# This will be replaced with module imports as we complete the refactoring
if __name__ == "__main__":
    # Import and run the original main function
    # This preserves all existing functionality
    import importlib.util
    original_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "kiwix_chat.py")
    if os.path.exists(original_file):
        spec = importlib.util.spec_from_file_location("kiwix_chat_original", original_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.exit(module.main())
    else:
        print("Error: Original kiwix_chat.py not found. Please ensure the file exists.")
        sys.exit(1)

