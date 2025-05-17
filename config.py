import torch
import os
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings with environment variable overrides."""
    
    # --- Base Paths ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")
    CRAWLED_DATA_DIR: str = os.path.join(BASE_DIR, "data_unpreprocessed")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data_processed")
    
    # --- Model Configuration ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    LLM_MODEL_NAME: str = "Qwen/Qwen1.5-1.8B-Chat"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- Artifact Storage Paths ---
    FAISS_INDEX_PATH: str = None  # Will be set dynamically 
    GRAPH_PATH: str = None        # Will be set dynamically
    DOC_STORE_PATH: str = None    # Will be set dynamically
    
    # --- KAGBuilder Configuration ---
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    
    # --- KAGSolver Configuration ---
    TOP_K_RETRIEVAL: int = 2
    MAX_REASONING_STEPS: int = 4
    
    class Config:
        env_prefix = ""  # No prefix for environment variables
        env_file = ".env"  # Read from .env file if it exists
        
    def setup(self):
        """Create necessary directories and initialize dynamic paths."""
        # Set dynamic paths that depend on ARTIFACTS_DIR
        if self.FAISS_INDEX_PATH is None:
            self.FAISS_INDEX_PATH = os.path.join(self.ARTIFACTS_DIR, "my_faiss_index.index")
        if self.GRAPH_PATH is None:
            self.GRAPH_PATH = os.path.join(self.ARTIFACTS_DIR, "my_knowledge_graph.gml")
        if self.DOC_STORE_PATH is None:
            self.DOC_STORE_PATH = os.path.join(self.ARTIFACTS_DIR, "doc_store.json")
            
        # Create directories
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.CRAWLED_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        
        
settings = Settings()
settings.setup()
