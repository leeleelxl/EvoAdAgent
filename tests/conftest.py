"""Pytest bootstrapping.

Works around a macOS OpenMP conflict: faiss and torch (transitively pulled in
by langchain) both link libomp; having two runtimes loaded aborts the process.
Setting KMP_DUPLICATE_LIB_OK=TRUE before any import is the documented workaround.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
