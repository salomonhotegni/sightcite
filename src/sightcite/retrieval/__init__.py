"""Text and visual retrieval."""

from sightcite.retrieval.embeddings import (
    BGE_QUERY_INSTRUCTION,
    DEFAULT_BGE_MODEL,
    BgeTextEmbedder,
    TextEmbedder,
)
from sightcite.retrieval.index import TextVectorIndex
from sightcite.retrieval.models import SearchResult
from sightcite.retrieval.text import TextRetriever

__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "DEFAULT_BGE_MODEL",
    "BgeTextEmbedder",
    "SearchResult",
    "TextEmbedder",
    "TextRetriever",
    "TextVectorIndex",
]
