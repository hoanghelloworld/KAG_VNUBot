from together import Together
import os 
from config import settings
import time

last_api_call_time = 0
MIN_API_CALL_INTERVAL = 15


client = Together(api_key=settings.TOGETHER_API_KEY)

def get_together_response(prompt, system_prompt= "", max_tokens=1000):
    global last_api_call_time
    
    current_time = time.time()
    elapsed_time_since_last_call = current_time - last_api_call_time
    
    if elapsed_time_since_last_call < MIN_API_CALL_INTERVAL:
        wait_time = MIN_API_CALL_INTERVAL - elapsed_time_since_last_call
        time.sleep(wait_time)
        
    response = client.chat.completions.create(
        model=settings.TOGETHER_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],  
        max_tokens=max_tokens,
        
    )
    last_api_call_time = time.time() # Cập nhật thời điểm gọi API cuối cùng
    return response.choices[0].message.content

if __name__ == "__main__":
    start_time = time.time()
    prompt = "What is the capital of France?"
    response = get_together_response(prompt) 
    print(response)
    
    prompt = "What is the capital of Vietnam?"
    response = get_together_response(prompt)
    print(response)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
