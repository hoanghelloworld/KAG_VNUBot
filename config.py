import torch
import os
from typing import Optional, Any, Dict
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

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
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" 
    
    # --- Artifact Storage Paths ---
    FAISS_INDEX_PATH: str = os.path.join(ARTIFACTS_DIR, "my_faiss_index.index")
    GRAPH_PATH: str = os.path.join(ARTIFACTS_DIR, "my_knowledge_graph.gml")
    DOC_STORE_PATH: str = os.path.join(ARTIFACTS_DIR, "doc_store.json")
    
    # --- KAGBuilder Configuration ---
    # CHUNK_SIZE: int = 300
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 30
    
    # --- KAGSolver Configuration ---
    TOP_K_RETRIEVAL: int = 5
    MAX_REASONING_STEPS: int = 6
    
    # --- Together API ---
    TOGETHER_API_KEY: Optional[str] = None
    TOGETHER_MODEL_NAME: Optional[str] = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
    # TOGETHER_MODEL_NAME: Optional[str] = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
    
    # --- Continue building KAG ---
    CONTINUE_BUILDING_KAG: bool = True
    
    
    class Config:
        env_prefix = ""  # No prefix for environment variables
        env_file = ".env"  # Read from .env file if it exists
        extra = "ignore"  # Ignore extra attributes like together_api
    

settings = Settings()
prompt_manager = PromptManager(settings.PROMPT_DIR)

# For debugging purposes only
if __name__ == "__main__":
    print(settings.TOGETHER_MODEL_NAME)
    print(settings.TOGETHER_API_KEY)
    print(settings.FAISS_INDEX_PATH)
    print(prompt_manager.sys_prompt_reasoning_agent)
    
