#!/usr/bin/env python3
"""Test script to verify RAG embedding models are working correctly."""

import sys
import os

def test_embedding_model():
    """Test that the embedding model loads and generates embeddings."""
    print("=" * 70)
    print("TEST 1: Embedding Model Loading")
    print("=" * 70)
    
    try:
        from kiwix_chat.rag.embeddings import initialize_embedding_model, generate_embedding, get_embedding_dimension
        
        print("\n[1.1] Initializing embedding model...")
        model = initialize_embedding_model()
        print(f"✓ Model loaded: {type(model).__name__}")
        
        print("\n[1.2] Getting embedding dimension...")
        dim = get_embedding_dimension()
        print(f"✓ Embedding dimension: {dim}")
        
        print("\n[1.3] Testing query embedding generation...")
        test_query = "What is the Treaty of Waitangi?"
        query_embedding = generate_embedding(test_query, is_query=True)
        print(f"✓ Query embedding generated: {len(query_embedding)} dimensions")
        print(f"  First 5 values: {query_embedding[:5]}")
        
        print("\n[1.4] Testing passage embedding generation...")
        test_passage = "The Treaty of Waitangi is a treaty first signed on 6 February 1840."
        passage_embedding = generate_embedding(test_passage, is_query=False)
        print(f"✓ Passage embedding generated: {len(passage_embedding)} dimensions")
        print(f"  First 5 values: {passage_embedding[:5]}")
        
        return True
    except ImportError as e:
        print(f"✗ ERROR: Missing dependencies: {e}")
        print("  Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"✗ ERROR: Failed to load embedding model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_store():
    """Test that vector store can be accessed."""
    print("\n" + "=" * 70)
    print("TEST 2: Vector Store Access")
    print("=" * 70)
    
    try:
        from kiwix_chat.kiwix.client import get_current_zim_file_path
        from kiwix_chat.rag.vector_store import get_vector_store, is_indexed
        
        print("\n[2.1] Checking for ZIM file...")
        zim_file_path = get_current_zim_file_path()
        if not zim_file_path:
            print("✗ No ZIM file found")
            print("  Place a .zim file in the app folder")
            return False
        
        print(f"✓ ZIM file found: {os.path.basename(zim_file_path)}")
        
        print("\n[2.2] Checking if index exists...")
        if not is_indexed(zim_file_path):
            print("✗ No vector index found")
            print("  Build index with: python3 kiwix_chat.py --build-index")
            return False
        
        print("✓ Vector index exists")
        
        print("\n[2.3] Loading vector store...")
        vector_store = get_vector_store(zim_file_path)
        print(f"✓ Vector store loaded: {type(vector_store).__name__}")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rag_retrieval():
    """Test that RAG retrieval works end-to-end."""
    print("\n" + "=" * 70)
    print("TEST 3: RAG Retrieval")
    print("=" * 70)
    
    try:
        from kiwix_chat.kiwix.client import get_current_zim_file_path
        from kiwix_chat.rag.vector_store import get_vector_store, is_indexed
        from kiwix_chat.rag.retriever import RAGRetriever
        
        print("\n[3.1] Setting up retriever...")
        zim_file_path = get_current_zim_file_path()
        if not zim_file_path or not is_indexed(zim_file_path):
            print("✗ ZIM file or index not available")
            return False
        
        vector_store = get_vector_store(zim_file_path)
        retriever = RAGRetriever(zim_file_path, vector_store)
        print("✓ Retriever created")
        
        print("\n[3.2] Testing retrieval with sample query...")
        test_query = "Treaty of Waitangi"
        chunks = retriever.retrieve(
            query=test_query,
            top_k=3,
            use_hybrid=True
        )
        
        if not chunks:
            print("✗ No chunks retrieved")
            return False
        
        print(f"✓ Retrieved {len(chunks)} chunks")
        for i, chunk in enumerate(chunks[:3], 1):
            print(f"  [{i}] Article: {chunk.article_title}")
            print(f"      Text preview: {chunk.text[:80]}...")
        
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("RAG EMBEDDING SYSTEM TEST")
    print("=" * 70)
    
    results = []
    
    # Test 1: Embedding model
    results.append(("Embedding Model", test_embedding_model()))
    
    # Test 2: Vector store
    results.append(("Vector Store", test_vector_store()))
    
    # Test 3: RAG retrieval
    results.append(("RAG Retrieval", test_rag_retrieval()))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
        return 0
    else:
        print("SOME TESTS FAILED ✗")
        return 1


if __name__ == "__main__":
    sys.exit(main())

