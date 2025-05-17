# llm_utils.py

import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer # Không cần cho API LLM
from sentence_transformers import SentenceTransformer
import config # Import cấu hình chung
from together import Together # Import thư viện Together

# --- Khởi tạo Model một lần và tái sử dụng ---
# _llm_tokenizer = None # Không cần cho API LLM
# _llm_model = None # Không cần cho API LLM
_embedding_model = None

# Khởi tạo Together client
if config.TOGETHER_API_KEY:
    try:
        _together_client = Together(api_key=config.TOGETHER_API_KEY)
    except Exception as e:
        _together_client = None
        print(f"Lỗi khi khởi tạo Together client: {e}. Hãy chắc chắn API key hợp lệ.")
else:
    _together_client = None
    print("Cảnh báo: TOGETHER_API_KEY không được tìm thấy trong config. Các cuộc gọi API sẽ thất bại.")

def get_llm_tokenizer():
    """
    Lấy tokenizer. Hàm này có thể vẫn cần thiết nếu bạn cần đếm token
    hoặc xử lý văn bản trước khi gửi đến API.
    Nếu không, nó có thể được loại bỏ hoặc trả về None.
    Hiện tại, chúng ta sẽ giữ lại cấu trúc nhưng nó không được sử dụng để tạo response qua API.
    """
    # from transformers import AutoTokenizer
    # global _llm_tokenizer
    # if _llm_tokenizer is None:
    #     try:
    #         # Bạn có thể muốn load một tokenizer tương thích với model API nếu cần
    #         # _llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME) # LLM_MODEL_NAME giờ là tên model API
    #         print(f"Tokenizer cho model API ({config.LLM_MODEL_NAME}) không được load mặc định khi dùng API.")
    #         pass # Không load tokenizer cho LLM khi dùng API
    #     except Exception as e:
    #         print(f"Không thể load tokenizer cho {config.LLM_MODEL_NAME} (có thể không cần thiết khi dùng API): {e}")
    # return _llm_tokenizer
    print("get_llm_tokenizer: Không áp dụng trực tiếp khi dùng API cho LLM response.")
    return None # Hoặc một tokenizer nếu cần cho các tác vụ khác

def get_llm_model():
    """
    Lấy model LLM. Hàm này không còn cần thiết khi sử dụng API.
    """
    # global _llm_model
    # if _llm_model is None:
    #     # Model loading logic will be removed as we are using API
    #     pass
    # return _llm_model
    print("get_llm_model: Không áp dụng khi dùng API cho LLM response.")
    return None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"LLM_UTILS: Loading embedding model: {config.EMBEDDING_MODEL_NAME} onto {config.DEVICE}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
    return _embedding_model

# --- Helper Functions ---
def get_llm_response(prompt_text, max_new_tokens=250, system_message="You are a helpful assistant.", stop_sequences=None):
    """
    Gửi prompt đến LLM API của Together và nhận phản hồi.
    """
    if not _together_client:
        error_msg = "Lỗi: Together client chưa được khởi tạo. Kiểm tra TOGETHER_API_KEY trong config."
        print(error_msg)
        return error_msg

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt_text})

    try:
        response = _together_client.chat.completions.create(
            model=config.LLM_MODEL_NAME,  # Sử dụng tên model API từ config
            messages=messages,
            max_tokens=max_new_tokens,
            stop=stop_sequences if stop_sequences else None
        )
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"Lỗi khi gọi Together API: {e}"
        print(error_msg)
        return error_msg

def get_embeddings(texts):
    """
    Tạo vector embedding cho danh sách các đoạn văn bản.
    """
    model = get_embedding_model()
    # Xử lý batching nếu danh sách texts quá lớn để tránh OOM
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, device=config.DEVICE)
        all_embeddings.append(batch_embeddings)
    
    if len(all_embeddings) == 1:
        return all_embeddings[0]
    else:
        return torch.cat(all_embeddings, dim=0)

# Thêm các hàm tiện ích liên quan đến LLM khác nếu cần
def calculate_token_probabilities(prompt_text, next_tokens=5):
    """
    Tính toán xác suất token. Hàm này có thể không hoạt động với API
    hoặc cần được viết lại hoàn toàn nếu API hỗ trợ.
    """
    # tokenizer = get_llm_tokenizer()
    # model = get_llm_model()
    # if not tokenizer or not model:
    #     return "Tokenizer hoặc model không có sẵn (không áp dụng cho API)."
    # inputs = tokenizer(prompt_text, return_tensors="pt").to(config.DEVICE)
    # # ... (logic cũ) ...
    print("calculate_token_probabilities: Không áp dụng trực tiếp hoặc cần viết lại cho API.")
    return {}

def check_context_length(text, model_name=None):
    """
    Kiểm tra độ dài context của văn bản so với giới hạn của model.
    
    Args:
        text: Văn bản cần kiểm tra
        model_name: Tên model để xác định giới hạn (nếu None, sẽ dùng model mặc định)
        
    Returns:
        tuple: (số tokens, giới hạn của model, có vượt quá không)
    """
    if model_name is None:
        model_name = config.LLM_MODEL_NAME
        
    tokenizer = get_llm_tokenizer()
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
    # Xác định giới hạn của model
    model_limits = {
        "gpt-3.5-turbo": 4096,
        "gpt-4": 8192,
        "phi-2": 2048,
        "llama2-7b": 4096,
        "llama2-13b": 4096,
        "llama3-8b": 8192,
        "mistral-7b": 8192,
        "mistral-8x7b": 32768,
    }
    
    limit = model_limits.get(model_name.lower(), 2048)  # Mặc định là 2048 nếu không biết model
    
    return token_count, limit, token_count > limit

def truncate_to_max_tokens(text, max_tokens=None, truncation_method="end"):
    """
    Cắt bớt văn bản để đảm bảo không vượt quá số token tối đa.
    
    Args:
        text: Văn bản cần cắt
        max_tokens: Số token tối đa (mặc định lấy theo giới hạn của model)
        truncation_method: Phương pháp cắt ("start", "end", hoặc "middle")
        
    Returns:
        Văn bản đã được cắt bớt
    """
    tokenizer = get_llm_tokenizer()
    tokens = tokenizer.encode(text)
    
    if max_tokens is None:
        _, max_tokens, _ = check_context_length(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    if truncation_method == "end":
        # Giữ phần đầu, cắt phần cuối
        truncated_tokens = tokens[:max_tokens]
    elif truncation_method == "start":
        # Giữ phần cuối, cắt phần đầu
        truncated_tokens = tokens[-max_tokens:]
    elif truncation_method == "middle":
        # Giữ đầu và cuối, cắt phần giữa
        half = max_tokens // 2
        truncated_tokens = tokens[:half] + tokens[-half:]
    else:
        raise ValueError(f"Phương pháp cắt không hợp lệ: {truncation_method}")
    
    return tokenizer.decode(truncated_tokens)

def chunk_text(text, max_chunk_size=1024, overlap=100):
    """
    Chia văn bản thành các đoạn nhỏ hơn với độ chồng lấn.
    
    Args:
        text: Văn bản cần chia
        max_chunk_size: Kích thước tối đa của mỗi đoạn (tính bằng token)
        overlap: Số token chồng lấn giữa các đoạn
        
    Returns:
        Danh sách các đoạn văn bản
    """
    tokenizer = get_llm_tokenizer()
    tokens = tokenizer.encode(text)
    
    if len(tokens) <= max_chunk_size:
        return [text]
    
    chunks = []
    for i in range(0, len(tokens), max_chunk_size - overlap):
        chunk_tokens = tokens[i:i + max_chunk_size]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        
        if i + max_chunk_size >= len(tokens):
            break
    
    return chunks
