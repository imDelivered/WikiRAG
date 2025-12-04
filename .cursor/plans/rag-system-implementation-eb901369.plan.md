<!-- eb901369-b636-4f1b-bd2d-1d2e326be5d0 ecd87273-fb03-474e-b238-d4020b857c52 -->
# Fix RAG System Issues

## Problems Identified

1. **RAG Not Being Used**: System falls back to keyword search because no index exists

   - No user notification when index is missing
   - Silent fallback makes it unclear why RAG isn't working

2. **Broken Topic Extraction**: LLM extracts completely wrong topics

   - Query: "Treaty of Waitangi" → Extracts: "PHOTOSYNTHESIS", "CHLOROPHYLL", "Plant", "MAUI"
   - This suggests the LLM topic extraction prompt or model is failing

3. **No Auto-Index Building**: User must manually run `--build-index`

   - Should offer to build index automatically or provide clear instructions

## Solutions

### 1. Improve RAG Index Detection & Feedback

**File**: `kiwix_chat.py` - `intelligent_wiki_fetch()`

- Add detailed logging when index check fails
- Show why RAG isn't being used (no index, dependencies missing, etc.)
- Add informational message suggesting user build index

**Changes**:

```python
# In intelligent_wiki_fetch(), add better error messages:
if zim_file_path and is_indexed(zim_file_path):
    # Use RAG
else:
    if zim_file_path:
        print(f"[rag] No index found for {os.path.basename(zim_file_path)}. Build index with: --build-index", file=sys.stderr)
    else:
        print(f"[rag] No ZIM file found. RAG unavailable.", file=sys.stderr)
```

### 2. Fix LLM Topic Extraction

**File**: `kiwix_chat.py` - `extract_wiki_topics_from_query()`

- The prompt may be too vague or the model is hallucinating
- Add validation to check if extracted topics make sense
- Improve prompt with better examples and constraints
- Add fallback validation against actual ZIM content

**Changes**:

- Enhance prompt with explicit examples of good vs bad topic extraction
- Add validation step: check if topics are relevant to query
- If validation fails, use simpler keyword-based extraction

### 3. Add Optional Auto-Index Building

**File**: `kiwix_chat.py` - `intelligent_wiki_fetch()` or `main()`

- Option 1: Prompt user on first use if index missing
- Option 2: Auto-build index in background (with user confirmation)
- Option 3: Add `--auto-build-index` flag

**Recommendation**: Add informational message with clear instructions, don't auto-build (too slow/large)

### 4. Add RAG Diagnostic Command

**File**: `kiwix_chat.py` - `parse_args()` and `main()`

- Add `--rag-status` command to show:
  - ZIM file path
  - Index status
  - Index size (chunks)
  - Embedding model loaded
  - Reranker status

### 5. Improve Error Handling

**Files**: `kiwix_chat/rag/embeddings.py`, `kiwix_chat/rag/retriever.py`

- Better error messages when embedding model fails to load
- Check if BGE models are actually downloaded
- Validate embedding generation works

## Implementation Priority

1. **High**: Add better logging/feedback for missing index
2. **High**: Fix topic extraction (it's completely broken)
3. **Medium**: Add diagnostic command
4. **Low**: Optional auto-index building

## Testing

After fixes, verify:

- RAG is used when index exists
- Clear messages when index is missing
- Topic extraction produces relevant topics
- Embedding models load correctly

### To-dos

- [ ] Add better logging and user feedback when RAG index is missing or unavailable
- [ ] Fix LLM topic extraction - add validation and improve prompt to prevent wrong topics like PHOTOSYNTHESIS for Treaty of Waitangi
- [ ] Add --rag-status command to show RAG system status (index, models, etc.)
- [ ] Improve error handling for embedding model loading and RAG operations