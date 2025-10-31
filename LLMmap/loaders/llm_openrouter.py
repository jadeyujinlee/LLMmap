## Ignore this file. Forgot what I was doing here. 


# import os
# from openai import OpenAI

# class LLM_OpenRouter:
#     """OpenRouter LLM loader using OpenAI-compatible API"""
    
#     def __init__(self, model_name, **kwargs):
#         self.model_name = model_name
#         self.api_key = os.getenv('OPENROUTER_API_KEY')
        
#         if not self.api_key:
#             raise ValueError("OPENROUTER_API_KEY environment variable not set")
        
#         # OpenRouter uses OpenAI SDK with custom base URL
#         self.client = OpenAI(
#             base_url="https://openrouter.ai/api/v1",
#             api_key=self.api_key,
#         )
        
#         # Default parameters
#         self.temperature = kwargs.get('temperature', 0.7)
#         self.max_tokens = kwargs.get('max_tokens', 512)
    
#     def query(self, prompt, system_prompt=None, **kwargs):
#         """Query the LLM via OpenRouter"""
#         messages = []
        
#         if system_prompt:
#             messages.append({"role": "system", "content": system_prompt})
        
#         messages.append({"role": "user", "content": prompt})
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=messages,
#                 temperature=kwargs.get('temperature', self.temperature),
#                 max_tokens=kwargs.get('max_tokens', self.max_tokens),
#             )
#             return response.choices[0].message.content
        
#         except Exception as e:
#             print(f"Error querying {self.model_name}: {e}")
#             return f"ERROR: {str(e)}"
    
#     def __call__(self, prompt, **kwargs):
#         """Make the class callable"""
#         return self.query(prompt, **kwargs)