"""RAG retriever for semantic search and retrieval."""

import sys
from typing import List, Optional

from kiwix_chat.models import Chunk
from kiwix_chat.rag.embeddings import generate_embedding
from kiwix_chat.rag.vector_store import VectorStore


class RAGRetriever:
    """Retriever for RAG system using vector embeddings."""
    
    def __init__(self, zim_file_path: str, vector_store: VectorStore):
        """Initialize RAG retriever.
        
        Args:
            zim_file_path: Path to ZIM file
            vector_store: VectorStore instance
        """
        self.zim_file_path = zim_file_path
        self.vector_store = vector_store
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        filter_metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            use_hybrid: Whether to use hybrid search (semantic + keyword)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of Chunk objects, sorted by relevance
        """
        if not query or not query.strip():
            return []
        
        # Generate query embedding (is_query=True for BGE models)
        try:
            query_embedding = generate_embedding(query, is_query=True)
            if not query_embedding or len(query_embedding) == 0:
                print(f"[rag] ERROR: Generated empty embedding", file=sys.stderr)
                return []
        except ImportError as e:
            print(f"[rag] ERROR: Embedding dependencies not available: {e}", file=sys.stderr)
            print(f"[rag] Install with: pip install sentence-transformers", file=sys.stderr)
            return []
        except Exception as e:
            print(f"[rag] ERROR: Failed to generate embedding: {e}", file=sys.stderr)
            import traceback
            if "--debug" in sys.argv or "-v" in sys.argv:
                traceback.print_exc(file=sys.stderr)
            return []
        
        # Perform search
        if use_hybrid:
            chunks = self.vector_store.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
        else:
            chunks = self.vector_store.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_metadata=filter_metadata
            )
        
        # Deduplicate by article (keep best chunk per article)
        deduplicated = self._deduplicate_by_article(chunks)
        
        return deduplicated[:top_k]
    
    def _deduplicate_by_article(self, chunks: List[Chunk]) -> List[Chunk]:
        """Deduplicate chunks by article, keeping the first (best) chunk per article.
        
        Args:
            chunks: List of chunks (assumed sorted by relevance)
            
        Returns:
            Deduplicated list
        """
        seen_articles = set()
        deduplicated = []
        
        for chunk in chunks:
            article_key = (chunk.article_title, chunk.href)
            if article_key not in seen_articles:
                seen_articles.add(article_key)
                deduplicated.append(chunk)
        
        return deduplicated
    
    def retrieve_with_reranking(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[Chunk]:
        """Retrieve chunks with cross-encoder reranking (same as Perplexica uses).
        
        Uses BGE reranker model for better relevance scoring.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters
            
        Returns:
            List of Chunk objects, sorted by relevance
        """
        # Get more candidates for reranking
        candidates = self.retrieve(query, top_k=top_k * 3, use_hybrid=True, filter_metadata=filter_metadata)
        
        if not candidates or len(candidates) <= top_k:
            return candidates[:top_k]
        
        # Rerank using cross-encoder
        try:
            from sentence_transformers import CrossEncoder
            
            # Use BGE reranker (same as Perplexica)
            reranker_model_name = "BAAI/bge-reranker-base"
            
            # Lazy load reranker (class-level cache)
            if not hasattr(RAGRetriever, '_reranker'):
                print(f"[rag] Loading reranker model: {reranker_model_name}", file=sys.stderr)
                print(f"[rag] First-time download may take a few minutes (~130MB)...", file=sys.stderr)
                try:
                    RAGRetriever._reranker = CrossEncoder(reranker_model_name)
                    print(f"[rag] Reranker loaded successfully", file=sys.stderr)
                except OSError as e:
                    print(f"[rag] ERROR: Failed to load reranker model: {e}", file=sys.stderr)
                    print(f"[rag] Check your internet connection (first download) or model cache.", file=sys.stderr)
                    raise
                except Exception as e:
                    print(f"[rag] ERROR: Reranker initialization failed: {e}", file=sys.stderr)
                    raise
            
            reranker = RAGRetriever._reranker
            
            # Prepare pairs for reranking: (query, chunk_text)
            pairs = [(query, chunk.text) for chunk in candidates]
            
            # Get reranking scores
            scores = reranker.predict(pairs)
            
            # Sort by score (higher is better)
            scored_chunks = list(zip(scores, candidates))
            scored_chunks.sort(key=lambda x: -x[0])  # Sort descending
            
            # Return top_k
            return [chunk for _, chunk in scored_chunks[:top_k]]
        except ImportError:
            print("[rag] Warning: CrossEncoder not available. Install with: pip install sentence-transformers[cross-encoder]", file=sys.stderr)
            # Fallback to regular retrieval
            return candidates[:top_k]
        except Exception as e:
            print(f"[rag] Warning: Reranking failed: {e}. Using regular retrieval.", file=sys.stderr)
            return candidates[:top_k]

