"""Data models for Kiwix RAG system."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class ModelPlatform(Enum):
    """Platform types for model execution."""
    OLLAMA = "ollama"
    AUTO = "auto"


@dataclass
class Message:
    """Chat message with role and content."""
    role: str
    content: str


@dataclass
class ArticleLink:
    """Link to a Kiwix article."""
    text: str
    href: str  # Kiwix relative path


@dataclass
class Chunk:
    """Chunk of text from an article for RAG retrieval."""
    text: str
    article_title: str
    href: str
    chunk_idx: int
    total_chunks: int = 1
    content_type: str = "Kiwix content"

