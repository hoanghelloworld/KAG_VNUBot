# llm_utils.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import config # Import cấu hình chung

# --- Khởi tạo Model một lần và tái sử dụng ---
_llm_tokenizer = None
_llm_model = None
_embedding_model = None

def get_llm_tokenizer():
    global _llm_tokenizer
    if _llm_tokenizer is None:
        print(f"LLM_UTILS: Loading LLM tokenizer: {config.LLM_MODEL_NAME}")
        _llm_tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME, trust_remote_code=True)
    return _llm_tokenizer

def get_llm_model():
    global _llm_model
    if _llm_model is None:
        print(f"LLM_UTILS: Loading LLM model: {config.LLM_MODEL_NAME} onto {config.DEVICE}")
        tokenizer = get_llm_tokenizer() # Đảm bảo tokenizer đã được tải
        if config.DEVICE == "cuda":
            _llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        else:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                config.LLM_MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(config.DEVICE)
    return _llm_model

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"LLM_UTILS: Loading embedding model: {config.EMBEDDING_MODEL_NAME} onto {config.DEVICE}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device=config.DEVICE)
    return _embedding_model

# --- Helper Functions ---
def get_llm_response(prompt_text, max_new_tokens=250, system_message="You are a helpful assistant.", stop_sequences=None):
    """
    Gửi prompt đến LLM và nhận phản hồi.
    """
    tokenizer = get_llm_tokenizer()
    model = get_llm_model()

    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(config.DEVICE)

    # Xử lý stop_sequences (nếu cần thiết và phức tạp hơn, có thể tạo custom StoppingCriteria)
    # Ví dụ đơn giản về stop_sequences (có thể không hoạt động hoàn hảo với mọi model/tokenizer)
    stopping_criteria = []
    # if stop_sequences:
    #     from transformers import StoppingCriteria, StoppingCriteriaList
    #     class StopOnTokens(StoppingCriteria):
    #         def __init__(self, stop_token_ids):
    #             super().__init__()
    #             self.stop_token_ids = stop_token_ids
    #         def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
    #             for stop_ids in self.stop_token_ids:
    #                 if torch.equal(input_ids[0][-len(stop_ids):], stop_ids.to(input_ids.device)):
    #                     return True
    #             return False
    #     stop_token_ids_list = [tokenizer.encode(s, add_special_tokens=False, return_tensors="pt")[0] for s in stop_sequences]
    #     stopping_criteria.append(StopOnTokens(stop_token_ids_list))
    
    # Sử dụng pad_token_id là eos_token_id nếu có, nếu không thì là pad_token_id của tokenizer
    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if pad_token_id is None and hasattr(tokenizer, 'pad_token_id_from_model_config'): # Một số model mới có thể cần cách này
         pad_token_id = tokenizer.pad_token_id_from_model_config

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id, # Quan trọng
        # stopping_criteria=StoppingCriteriaList(stopping_criteria) if stopping_criteria else None
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Xử lý stop_sequences thủ công nếu generate không hỗ trợ tốt
    if stop_sequences:
        for seq in stop_sequences:
            if seq in response:
                response = response.split(seq)[0]
                break
    return response.strip()


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
    Tính xác suất của các token tiếp theo dựa trên prompt đã cho.
    
    Args:
        prompt_text: Đoạn văn bản đầu vào
        next_tokens: Số lượng token có xác suất cao nhất muốn lấy
        
    Returns:
        Danh sách các tuple (token, xác suất)
    """
    tokenizer = get_llm_tokenizer()
    model = get_llm_model()
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(config.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Lấy logits cho token cuối cùng
    logits = outputs.logits[0, -1, :]
    probabilities = torch.nn.functional.softmax(logits, dim=0)
    
    # Lấy top-k tokens có xác suất cao nhất
    topk_probs, topk_indices = torch.topk(probabilities, next_tokens)
    
    results = []
    for i, (prob, idx) in enumerate(zip(topk_probs, topk_indices)):
        token = tokenizer.decode([idx])
        results.append((token, prob.item()))
        
    return results

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
