"""
DocuMind AI - Production-Grade RAG Knowledge Retrieval Backend Entrypoint
Backward-compatibility wrapper loading app.main:app
"""

import os
import sys
import importlib.util

# Ensure root directory is in sys.path
root_dir = os.path.dirname(os.path.abspath(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Load app package explicitly if shadowed by app.py
if "app" not in sys.modules or not hasattr(sys.modules["app"], "__path__"):
    app_init_path = os.path.join(root_dir, "app", "__init__.py")
    spec = importlib.util.spec_from_file_location("app", app_init_path, submodule_search_locations=[os.path.join(root_dir, "app")])
    app_pkg = importlib.util.module_from_spec(spec)
    sys.modules["app"] = app_pkg
    spec.loader.exec_module(app_pkg)

from app.main import app
from app.config.settings import settings

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
