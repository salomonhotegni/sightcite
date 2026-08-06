"""Text and visual retrieval."""

from sightcite.retrieval.clip import (
    DEFAULT_CLIP_MODEL,
    ClipVisualEmbedder,
)
from sightcite.retrieval.embeddings import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_BGE_MODEL,
    BgeTextEmbedder,
    TextEmbedder,
)
from sightcite.retrieval.index import TextVectorIndex
from sightcite.retrieval.models import SearchResult
from sightcite.retrieval.text import TextRetriever
from sightcite.retrieval.visual_embeddings import VisualEmbedder

__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "DEFAULT_BGE_MODEL",
    "DEFAULT_CLIP_MODEL",
    "BgeTextEmbedder",
    "ClipVisualEmbedder",
    "SearchResult",
    "TextEmbedder",
    "TextRetriever",
    "TextVectorIndex",
    "VisualEmbedder",
]
