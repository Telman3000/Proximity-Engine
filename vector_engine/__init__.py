"""DLS vector-search engine (CPU-only).

A scalable, compressed semantic-retrieval engine: representation model ->
vector storage -> indexing -> cascaded search, evaluated with confidence
intervals. Thematically connected to the beyond-proximity research project.
"""

# IMPORTANT (Windows): faiss MUST be imported before pandas/pyarrow, otherwise
# their conflicting native runtimes cause an access violation in faiss HNSW
# graph construction. Loading faiss first makes its OpenMP/runtime win.
try:  # pragma: no cover - environment guard
    import faiss  # noqa: F401
except Exception:  # faiss optional for pure data-prep commands
    pass

__version__ = "0.1.0"
