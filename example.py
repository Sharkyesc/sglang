from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="not-needed" 
)

stream = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "以森林里有一只小白兔开头编一个故事"}
    ],
    temperature=0.7,
    max_tokens=100,
    stream=True 
)

full_response = ""
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        content = chunk.choices[0].delta.content
        print(content, end="", flush=True)
        full_response += content
print("\n")

# 非流式输出（等全部生成完才输出）
"""
print("\n=== 非流式输出模式 ===")
response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "1+1=?"}
    ],
    max_tokens=100,
    temperature=0.7,
    stream=False
)
print(response.choices[0].message.content) """