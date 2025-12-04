# Kiwix Chat Refactoring Plan

## Current State
- **File**: `kiwix_chat.py` (5,253 lines)
- **Issue**: Single file contains all functionality (GUI, terminal, Kiwix client, LLM, prompts, etc.)

## Refactoring Strategy

### Phase 1: Extract Core Modules (COMPLETED)
✅ Created module structure:
- `kiwix_chat/models.py` - Data models (Message, ArticleLink, ModelPlatform)
- `kiwix_chat/config.py` - Configuration constants and loading
- `kiwix_chat/kiwix/parser.py` - HTML parsers
- `kiwix_chat/kiwix/client.py` - Kiwix HTTP client and article fetching
- `kiwix_chat/chat/ollama.py` - Ollama API integration

### Phase 2: Remaining Modules to Extract

#### High Priority (Large, self-contained):
1. **`kiwix_chat/kiwix/context.py`** (~800 lines)
   - `extract_wiki_topics_from_query()`
   - `intelligent_wiki_fetch()`
   - `recursive_context_augmentation()`
   - `detect_missing_context()`
   - `kiwix_fetch_multiple_articles()`
   - `filter_relevant_context()`

2. **`kiwix_chat/prompts.py`** (~400 lines)
   - `DETAILED_PROMPT` constant
   - `enhance_system_prompt()`
   - `enhance_reasoning_prompt()`

3. **`kiwix_chat/chat/builder.py`** (~200 lines)
   - `build_messages()`

4. **`kiwix_chat/ui/gui.py`** (~1,900 lines)
   - `KiwixRAGGUI` class

5. **`kiwix_chat/ui/terminal.py`** (~600 lines)
   - Terminal interface code from `main()`

6. **`kiwix_chat/utils.py`** (~300 lines)
   - `extract_hyperlinks()`
   - `detect_entities()`
   - `annotate_text_with_wiki_links()`
   - `show_wiki_popup()`
   - `make_terminal_link()`
   - `should_fetch_wiki_context()`
   - `validate_response_uses_context()`
   - `extract_wiki_context_from_history()`
   - `generate_response_with_regeneration()`
   - `check_ollama_gpu()`

7. **`kiwix_chat/main.py`** (~200 lines)
   - `parse_args()`
   - `main()` entry point

## Migration Strategy

### Step 1: Update Original File Imports
Update `kiwix_chat.py` to import from new modules where they exist:

```python
# At top of kiwix_chat.py
from kiwix_chat.models import Message, ArticleLink, ModelPlatform
from kiwix_chat.config import (
    DEFAULT_MODEL, KIWIX_BASE_URL, OLLAMA_CHAT_URL,
    detect_platform, load_model_config
)
from kiwix_chat.kiwix.parser import HTMLParserWithLinks, KiwixSearchParser
from kiwix_chat.kiwix.client import (
    kiwix_fetch_article, kiwix_search_first_href, http_get,
    set_global_zim_file_path
)
from kiwix_chat.chat.ollama import stream_chat, full_chat
```

### Step 2: Remove Extracted Code
Once imports work, remove the extracted functions from `kiwix_chat.py`.

### Step 3: Test Functionality
Run the application to ensure all features still work.

## Benefits After Refactoring

- **Maintainability**: Each module < 1000 lines
- **Testability**: Can test modules independently
- **Reusability**: Modules can be imported separately
- **Clarity**: Clear separation of concerns
- **Collaboration**: Reduced merge conflicts

## File Size Targets

| Module | Target Lines | Status |
|--------|--------------|--------|
| models.py | ~50 | ✅ Done |
| config.py | ~100 | ✅ Done |
| kiwix/parser.py | ~100 | ✅ Done |
| kiwix/client.py | ~400 | ✅ Done |
| kiwix/context.py | ~800 | ⏳ Pending |
| prompts.py | ~400 | ⏳ Pending |
| chat/builder.py | ~200 | ⏳ Pending |
| chat/ollama.py | ~150 | ✅ Done |
| ui/gui.py | ~1900 | ⏳ Pending |
| ui/terminal.py | ~600 | ⏳ Pending |
| utils.py | ~300 | ⏳ Pending |
| main.py | ~200 | ⏳ Pending |

## Next Steps

1. Complete Phase 2 module extraction
2. Update `kiwix_chat.py` to use new modules
3. Test all functionality
4. Remove duplicate code from original file
5. Update documentation

