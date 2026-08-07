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
from sightcite.retrieval.fusion import (
    DEFAULT_RRF_CONSTANT,
    reciprocal_rank_fusion,
)
from sightcite.retrieval.index import (
    TextVectorIndex,
    VisualVectorIndex,
)
from sightcite.retrieval.models import (
    SearchResult,
    VisualSearchResult,
)
from sightcite.retrieval.text import TextRetriever
from sightcite.retrieval.visual import VisualRetriever
from sightcite.retrieval.visual_embeddings import VisualEmbedder

__all__ = [
    "BGE_QUERY_INSTRUCTION",
    "DEFAULT_BGE_MODEL",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_RRF_CONSTANT",
    "BgeTextEmbedder",
    "ClipVisualEmbedder",
    "FusedPageResult",
    "SearchResult",
    "TextEmbedder",
    "TextRetriever",
    "TextVectorIndex",
    "VisualEmbedder",
    "VisualRetriever",
    "VisualSearchResult",
    "VisualVectorIndex",
    "reciprocal_rank_fusion",
]
