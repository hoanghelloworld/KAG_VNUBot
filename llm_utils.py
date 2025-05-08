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
    # TODO: Xử lý batching nếu danh sách texts quá lớn để tránh OOM
    # Ví dụ:
    # batch_size = 32
    # all_embeddings = []
    # for i in range(0, len(texts), batch_size):
    #     batch_texts = texts[i:i+batch_size]
    #     batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, device=config.DEVICE)
    #     all_embeddings.append(batch_embeddings)
    # return torch.cat(all_embeddings, dim=0)
    return model.encode(texts, convert_to_tensor=True, device=config.DEVICE)

# TODO: Thêm các hàm tiện ích liên quan đến LLM khác nếu cần
# Ví dụ: hàm tính toán xác suất token, hàm kiểm tra độ dài context, ...
