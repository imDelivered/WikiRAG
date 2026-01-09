
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

"""
GBNF Grammar Utilities for JSON Enforcement.

This module provides GBNF (Generative BNF) grammar definitions that force
llama-cpp-python models to output strictly valid JSON. This eliminates
issues like conversational filler, trailing commas, and Markdown code blocks.

GBNF is a constraint-based approach that restricts token generation at the
model level, making it impossible for the model to output anything other
than valid JSON.
"""

from typing import Optional

# Cache the grammar instance to avoid recreation
_json_grammar_instance = None


# =============================================================================
# GBNF Grammar Definition
# =============================================================================

# This grammar enforces strict JSON according to RFC 8259.
# It prevents:
# - Conversational filler before/after JSON
# - Markdown code blocks (```json ... ```)
# - Trailing commas in arrays/objects
# - Unquoted strings
# - Single quotes instead of double quotes

JSON_GRAMMAR = r'''
root ::= value

value ::= object | array | string | number | boolean | null

object ::= "{" ws ( pair ( "," ws pair )* )? ws "}"

pair ::= string ws ":" ws value

array ::= "[" ws ( value ( "," ws value )* )? ws "]"

string ::= "\"" chars "\""

chars ::= char*

char ::= [^"\\] | "\\" escape

escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode

unicode ::= "u" hex hex hex hex

hex ::= [0-9a-fA-F]

number ::= integer fraction? exponent?

integer ::= "-"? ( "0" | [1-9] [0-9]* )

fraction ::= "." [0-9]+

exponent ::= [eE] [+-]? [0-9]+

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \t\n\r]*
'''

# Stricter grammar that starts with object or array only (no primitives at root)
JSON_OBJECT_OR_ARRAY_GRAMMAR = r'''
root ::= object | array

value ::= object | array | string | number | boolean | null

object ::= "{" ws ( pair ( "," ws pair )* )? ws "}"

pair ::= string ws ":" ws value

array ::= "[" ws ( value ( "," ws value )* )? ws "]"

string ::= "\"" chars "\""

chars ::= char*

char ::= [^"\\] | "\\" escape

escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode

unicode ::= "u" hex hex hex hex

hex ::= [0-9a-fA-F]

number ::= integer fraction? exponent?

integer ::= "-"? ( "0" | [1-9] [0-9]* )

fraction ::= "." [0-9]+

exponent ::= [eE] [+-]? [0-9]+

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \t\n\r]*
'''


def get_json_grammar():
    """
    Get a LlamaGrammar object that enforces valid JSON output.
    
    Returns:
        LlamaGrammar: A compiled grammar object, or None if compilation fails.
        
    Notes:
        - The grammar is cached after first creation for performance.
        - If llama_cpp is not available or grammar compilation fails,
          returns None (caller should fall back to regex extraction).
    """
    global _json_grammar_instance
    
    if _json_grammar_instance is not None:
        return _json_grammar_instance
    
    try:
        from llama_cpp import LlamaGrammar
        _json_grammar_instance = LlamaGrammar.from_string(JSON_OBJECT_OR_ARRAY_GRAMMAR)
        return _json_grammar_instance
    except ImportError:
        # llama_cpp not installed
        return None
    except Exception as e:
        # Grammar compilation failed
        import sys
        print(f"[WARN] Failed to compile JSON grammar: {e}", file=sys.stderr)
        return None


def get_array_grammar():
    """
    Get a LlamaGrammar object that enforces JSON array output only.
    
    This is useful when you specifically expect an array response,
    like a list of suggestions or scores.
    
    Returns:
        LlamaGrammar: A compiled grammar object, or None if compilation fails.
    """
    array_only_grammar = r'''
root ::= array

value ::= object | array | string | number | boolean | null

object ::= "{" ws ( pair ( "," ws pair )* )? ws "}"

pair ::= string ws ":" ws value

array ::= "[" ws ( value ( "," ws value )* )? ws "]"

string ::= "\"" chars "\""

chars ::= char*

char ::= [^"\\] | "\\" escape

escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode

unicode ::= "u" hex hex hex hex

hex ::= [0-9a-fA-F]

number ::= integer fraction? exponent?

integer ::= "-"? ( "0" | [1-9] [0-9]* )

fraction ::= "." [0-9]+

exponent ::= [eE] [+-]? [0-9]+

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \t\n\r]*
'''
    try:
        from llama_cpp import LlamaGrammar
        return LlamaGrammar.from_string(array_only_grammar)
    except (ImportError, Exception):
        return None


def get_object_grammar():
    """
    Get a LlamaGrammar object that enforces JSON object output only.
    
    This is useful when you specifically expect an object response,
    like entity extraction results.
    
    Returns:
        LlamaGrammar: A compiled grammar object, or None if compilation fails.
    """
    object_only_grammar = r'''
root ::= object

value ::= object | array | string | number | boolean | null

object ::= "{" ws ( pair ( "," ws pair )* )? ws "}"

pair ::= string ws ":" ws value

array ::= "[" ws ( value ( "," ws value )* )? ws "]"

string ::= "\"" chars "\""

chars ::= char*

char ::= [^"\\] | "\\" escape

escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | unicode

unicode ::= "u" hex hex hex hex

hex ::= [0-9a-fA-F]

number ::= integer fraction? exponent?

integer ::= "-"? ( "0" | [1-9] [0-9]* )

fraction ::= "." [0-9]+

exponent ::= [eE] [+-]? [0-9]+

boolean ::= "true" | "false"

null ::= "null"

ws ::= [ \t\n\r]*
'''
    try:
        from llama_cpp import LlamaGrammar
        return LlamaGrammar.from_string(object_only_grammar)
    except (ImportError, Exception):
        return None