"""Scientific-document ingestion."""

from sightcite.ingestion.models import RenderedPage
from sightcite.ingestion.pdf import render_pdf

__all__ = ["RenderedPage", "render_pdf"]
