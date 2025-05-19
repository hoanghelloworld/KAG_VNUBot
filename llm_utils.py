import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from config import settings, prompt_manager
from together import Together
import os
from together_api_model import get_together_response

# Initialize models once and reuse
_llm_tokenizer = None
_llm_model = None
_embedding_model = None

def get_llm_tokenizer():
    global _llm_tokenizer
    if _llm_tokenizer is None:
        print(f"LLM_UTILS: Loading LLM tokenizer: {settings.LLM_MODEL_NAME}")
        _llm_tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME, trust_remote_code=True)
    return _llm_tokenizer

def get_llm_model():
    global _llm_model
    if _llm_model is None:
        print(f"LLM_UTILS: Loading LLM model: {settings.LLM_MODEL_NAME} onto {settings.DEVICE}")
        tokenizer = get_llm_tokenizer()
        if settings.DEVICE == "cuda":
            _llm_model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_NAME,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            )
        else:
            _llm_model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_NAME,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(settings.DEVICE)
    return _llm_model

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"LLM_UTILS: Loading embedding model: {settings.EMBEDDING_MODEL_NAME} onto {settings.DEVICE}")
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device=settings.DEVICE)
    return _embedding_model

def get_hf_llm_response(prompt_text, max_new_tokens=250, system_message="You are a helpful assistant.", stop_sequences=None):
    """
    Send prompt to LLM and get response.
    """
    tokenizer = get_llm_tokenizer()
    model = get_llm_model()

    messages = [{"role": "system", "content": system_message},
                {"role": "user", "content": prompt_text}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(settings.DEVICE)

    pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if pad_token_id is None and hasattr(tokenizer, 'pad_token_id_from_model_settings'): 
         pad_token_id = tokenizer.pad_token_id_from_model_settings

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    if stop_sequences:
        for seq in stop_sequences:
            if seq in response:
                response = response.split(seq)[0]
                break
    return response.strip()

def get_llm_response(prompt_text, max_new_tokens=3000, system_message="You are a helpful assistant.", stop_sequences=None):
    """
    Send prompt to LLM and get response.
    If TOGETHER_API_KEY exists in settings, use Together API.
    Otherwise, use local model.
    """
    try:
        if hasattr(settings, 'TOGETHER_API_KEY') and settings.TOGETHER_API_KEY:
            return get_together_response(prompt_text, system_message, max_new_tokens)
        else:
            return get_hf_llm_response(prompt_text, max_new_tokens, system_message, stop_sequences)
    except (ImportError, AttributeError):
        return get_hf_llm_response(prompt_text, max_new_tokens, system_message, stop_sequences)

def get_embeddings(texts):
    """
    Create embedding vectors for a list of text passages.
    """
    model = get_embedding_model()
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True, device=settings.DEVICE)
        all_embeddings.append(batch_embeddings)
    
    if len(all_embeddings) == 1:
        return all_embeddings[0]
    else:
        return torch.cat(all_embeddings, dim=0)

def calculate_token_probabilities(prompt_text, next_tokens=5):
    """
    Calculate probabilities of next tokens based on the given prompt.
    
    Args:
        prompt_text: Input text
        next_tokens: Number of highest probability tokens to retrieve
        
    Returns:
        List of tuples (token, probability)
    """
    tokenizer = get_llm_tokenizer()
    model = get_llm_model()
    
    inputs = tokenizer(prompt_text, return_tensors="pt").to(settings.DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits[0, -1, :]
    probabilities = torch.nn.functional.softmax(logits, dim=0)
    
    topk_probs, topk_indices = torch.topk(probabilities, next_tokens)
    
    results = []
    for i, (prob, idx) in enumerate(zip(topk_probs, topk_indices)):
        token = tokenizer.decode([idx])
        results.append((token, prob.item()))
        
    return results

def check_context_length(text, model_name=None):
    """
    Check the context length of text against the model limit.
    
    Args:
        text: Text to check
        model_name: Model name to determine the limit (if None, will use default model)
        
    Returns:
        tuple: (token count, model limit, exceeds limit)
    """
    if model_name is None:
        model_name = settings.LLM_MODEL_NAME
        
    tokenizer = get_llm_tokenizer()
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    
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
    
    limit = model_limits.get(model_name.lower(), 2048)
    
    return token_count, limit, token_count > limit

def truncate_to_max_tokens(text, max_tokens=None, truncation_method="end"):
    """
    Truncate text to ensure it doesn't exceed maximum token count.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum number of tokens (default uses model limit)
        truncation_method: Truncation method ("start", "end", or "middle")
        
    Returns:
        Truncated text
    """
    tokenizer = get_llm_tokenizer()
    tokens = tokenizer.encode(text)
    
    if max_tokens is None:
        _, max_tokens, _ = check_context_length(text)
    
    if len(tokens) <= max_tokens:
        return text
    
    if truncation_method == "end":
        truncated_tokens = tokens[:max_tokens]
    elif truncation_method == "start":
        truncated_tokens = tokens[-max_tokens:]
    elif truncation_method == "middle":
        half = max_tokens // 2
        truncated_tokens = tokens[:half] + tokens[-half:]
    else:
        raise ValueError(f"Invalid truncation method: {truncation_method}")
    
    return tokenizer.decode(truncated_tokens)

def chunk_text(text, max_chunk_size=1024, overlap=100):
    """
    Split text into smaller chunks with overlap.
    
    Args:
        text: Text to split
        max_chunk_size: Maximum size of each chunk (in tokens)
        overlap: Number of overlapping tokens between chunks
        
    Returns:
        List of text chunks
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

if __name__ == "__main__":
    print(get_llm_response(prompt_text="Cho tôi biết về đại học quốc gia hà nội ?", system_message=prompt_manager.sys_prompt_reasoning_agent, max_new_tokens=250)) 
    
