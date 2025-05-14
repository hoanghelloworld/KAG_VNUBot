# config.py

import torch
import os

# --- Đường dẫn Cơ sở ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
CRAWLED_DATA_DIR = os.path.join(BASE_DIR, "data_unpreprocessed")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data_processed")

# --- Cấu hình Model ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# TODO: Cho phép người dùng chọn model LLM và Embedding từ đây hoặc qua biến môi trường
# Ví dụ: LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen1.5-1.8B-Chat")
LLM_MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Đường dẫn Lưu trữ Artifacts ---
FAISS_INDEX_PATH = os.path.join(ARTIFACTS_DIR, "my_faiss_index.index")
GRAPH_PATH = os.path.join(ARTIFACTS_DIR, "my_knowledge_graph.gml")
DOC_STORE_PATH = os.path.join(ARTIFACTS_DIR, "doc_store.json")

# --- Cấu hình KAGBuilder ---
CHUNK_SIZE = 300
CHUNK_OVERLAP = 30

# --- Cấu hình KAGSolver ---
TOP_K_RETRIEVAL = 2
MAX_REASONING_STEPS = 4

# --- Cấu hình Crawler ---
# TODO: Thêm URL mục tiêu và các cấu hình khác cho crawler
# Ví dụ: TARGET_WEBSITE_URL = "https://example-education-site.com/courses"
#        MAX_PAGES_TO_CRAWL = 100

# --- Đảm bảo các thư mục tồn tại ---
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CRAWLED_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# TODO: Thêm các cấu hình chung khác nếu cần
# Ví dụ: API keys (nếu dùng API LLM trả phí), ngưỡng confidence cho trích xuất, ...
