import torch
import os
from typing import Optional, Any
from pydantic_settings import BaseSettings

class PromptManager:
    """Manages system prompts for different components of the application."""
    
    def __init__(self, prompt_dir: str):
        """
        Initialize the PromptManager with the directory containing prompt files.
        
        Args:
            prompt_dir: Path to the directory containing prompt files
        """
        self.prompt_dir = prompt_dir
        self._load_all_prompts()
    
    def _load_all_prompts(self):
        """Load all prompt files from the prompt directory as attributes."""
        if not os.path.exists(self.prompt_dir):
            os.makedirs(self.prompt_dir, exist_ok=True)
            
        for filename in os.listdir(self.prompt_dir):
            if filename.endswith(".txt"):
                attr_name = os.path.splitext(filename)[0]
                content = self._load_prompt(filename)
                setattr(self, attr_name, content)
    
    def _load_prompt(self, filename: str) -> str:
        """Load a prompt from a file."""
        filepath = os.path.join(self.prompt_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            print(f"Warning: Prompt file {filename} not found in {self.prompt_dir}")
            return ""

class Settings(BaseSettings):
    """Application settings with environment variable overrides."""
    
    # --- Base Paths ---
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    ARTIFACTS_DIR: str = os.path.join(BASE_DIR, "artifacts")
    CRAWLED_DATA_DIR: str = os.path.join(BASE_DIR, "data_unpreprocessed")
    PROCESSED_DATA_DIR: str = os.path.join(BASE_DIR, "data_processed")
    PROMPT_DIR: str = os.path.join(BASE_DIR, "prompt")
    
    # --- Model Configuration ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    LLM_MODEL_NAME: str = "Qwen/Qwen1.5-1.8B-Chat"
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # --- Artifact Storage Paths ---
    FAISS_INDEX_PATH: Optional[str] = ""   
    GRAPH_PATH: Optional[str] = ""        
    DOC_STORE_PATH: Optional[str] = ""    
    
    # --- KAGBuilder Configuration ---
    CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 30
    
    # --- KAGSolver Configuration ---
    TOP_K_RETRIEVAL: int = 2
    MAX_REASONING_STEPS: int = 4
    
    # --- Prompt Manager ---
    prompt_manager: Optional[Any] = None
    
    class Config:
        env_prefix = ""  # No prefix for environment variables
        env_file = ".env"  # Read from .env file if it exists
        extra = "ignore"  # Ignore extra attributes like together_api
        
    def setup(self):
        """Create necessary directories and initialize dynamic paths."""
        # Set dynamic paths that depend on ARTIFACTS_DIR
        self.FAISS_INDEX_PATH = os.path.join(self.ARTIFACTS_DIR, "my_faiss_index.index")
        self.GRAPH_PATH = os.path.join(self.ARTIFACTS_DIR, "my_knowledge_graph.gml")
        self.DOC_STORE_PATH = os.path.join(self.ARTIFACTS_DIR, "doc_store.json")
            
        # Create directories
        os.makedirs(self.ARTIFACTS_DIR, exist_ok=True)
        os.makedirs(self.CRAWLED_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(self.PROMPT_DIR, exist_ok=True)
        
        # Initialize prompt manager
        self.prompt_manager = PromptManager(self.PROMPT_DIR)
        
        
settings = Settings()
settings.setup()

# For debugging purposes only
if __name__ == "__main__":
    print(settings.prompt_manager.sys_prompt_reasoning_agent)
    print("Settings successfully loaded!") 
