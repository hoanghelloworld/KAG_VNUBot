from together import Together
import os 
from config import settings

client = Together(api_key=settings.TOGETHER_API_KEY)

def get_together_response(prompt, system_prompt= "", max_tokens=1000):
    response = client.chat.completions.create(
        model=settings.TOGETHER_MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],  
        max_tokens=max_tokens,
        
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    prompt = "What is the capital of France?"
    response = get_together_response(prompt)
    print(response)
