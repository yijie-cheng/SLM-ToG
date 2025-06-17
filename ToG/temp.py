import re
import openai
import time
import jsonlines  # 用於處理jsonl格式檔案
from openai import OpenAI

def run_llm(prompt, temperature, max_tokens, openai_api_keys, engine):
    if "gpt" not in engine.lower():
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        engine = client.models.list().data[0].id
    else:
        client = OpenAI(api_key=openai_api_keys)

    messages = [{"role":"system","content":"You are an AI assistant that helps people find information."}]
    message_prompt = {"role":"user","content":prompt}
    messages.append(message_prompt)

    f = 0
    while(f == 0):
        try:
            response = client.chat.completions.create(
                model=engine,
                messages = messages,
                temperature=temperature,
                max_tokens=max_tokens,
                frequency_penalty=0,
                presence_penalty=0,
            )
            result = response.choices[0].message.content
            f = 1
        except Exception as e:
            print(f"Error during request, retrying: {e}")
            time.sleep(2)
    return result

input_file = "./results/prunetool_results/gpt4.1-gpt4.1-cwq-ToG-log.txt"
jsonl_file = "gpt-4.1+Llama8b-cwq-ToG.jsonl"

# 讀取txt檔案內容
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

split_text = '''
------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''
entries = content.split(split_text)

# 打開並讀取jsonl檔案
with jsonlines.open(jsonl_file, mode='r') as reader:
    data = list(reader)

# 確保有足夠的entries和jsonl資料行數
if len(entries) != len(data):
    print(f"Warning: The number of entries ({len(entries)}) does not match the number of JSONL records ({len(data)}).")

# 將結果寫入jsonl檔案
for i, entry in enumerate(entries):
    print(i)
    prompts = re.findall(r"PROMPT:(.*?)RESULT:", entry, re.DOTALL)
    
    if prompts:
        prompt = prompts[-1].strip()  # 去除前後空格
        # result = run_llm(prompt, 0, 1024, '', "meta-llama/Meta-Llama-3-8B-Instruct")
        # result = run_llm(prompt, 0, 1024, "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", "gpt-4o-mini")
        result = run_llm(prompt, 0, 1024, '', "meta-llama/Meta-Llama-3-8B-Instruct")
        
        # 更新jsonl檔案中對應的result欄位
        data[i]["results"] = result
    
    else:
        print(f"No PROMPT and RESULT section found in entry {i + 1}.")

# 寫回修改後的jsonl檔案
with jsonlines.open(jsonl_file, mode='w') as writer:
    writer.write_all(data)

print("Results have been updated in the JSONL file.")
