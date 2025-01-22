import json
import argparse
import time
from openai import OpenAI
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int,
                        default=1024, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    args = parser.parse_args()

sep_string = '''
------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''

# 讀取文件內容
def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# 提取 [content] 和 [question] 的函數
def extract_contents(file_content):
    sections = file_content.split(sep_string)
    extracted_data = []

    for section in sections:
        # 找到包含 PROMPT 和 RESULT 的段落
        if 'PROMPT:' in section and 'RESULT:' in section:
            try:
                # 獲取所有 PROMPT 和 RESULT 的索引
                prompts = [i for i in range(len(section)) if section.startswith('PROMPT:', i)]
                results = [i for i in range(len(section)) if section.startswith('RESULT:', i)]

                if prompts and results:
                    # 取最後一個 PROMPT 和其對應的 RESULT
                    content_start = prompts[-1] + len('PROMPT:')
                    content_end = results[-1]
                    content = section[content_start:content_end].strip()

                    # 提取 [question]
                    question_start = section.find('question:')
                    if question_start != -1:
                        question_end = section.find('\n', question_start)
                        question = section[question_start + len('question:'):question_end].strip()
                    else:
                        question = ""

                    # 添加到結果列表中
                    extracted_data.append({
                        "question": question,
                        "gpt4omini_prompt": content
                    })
            except ValueError:
                # 如果索引出錯，跳過該段
                continue

    return extracted_data

# 實時更新單個數據到 JSON 文件
def append_to_json(data, output_path):
    with open(output_file_path, 'r+', encoding='utf-8') as json_file:
        try:
            file_data = json.load(json_file)
        except json.JSONDecodeError:
            file_data = []

        for existing_data in file_data:
            if existing_data["question"] == data["question"] and existing_data["gpt4omini_prompt"] == data["gpt4omini_prompt"]:
                existing_data.update(data)
                break
        else:
            file_data.append(data)

        json_file.seek(0)
        json.dump(file_data, json_file, ensure_ascii=False, indent=4)
        json_file.truncate()

# 調用 LLM 的函數
def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine):
    if "gpt" not in engine.lower():
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        engine = client.models.list().data[0].id
    else:
        client = OpenAI(api_key=opeani_api_keys)

    messages = [{"role": "system", "content": "You are an AI assistant that helps people find information."}]
    message_prompt = {"role": "user", "content": prompt}
    messages.append(message_prompt)

    f = 0
    while f == 0:
        try:
            response = client.chat.completions.create(
                model=engine,
                messages=messages,
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

# 主函數
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_length", type=int,
                        default=1024, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    args = parser.parse_args()

    input_file_path = "gpt-4o-mini-cwq-ToG-log.txt"  # 替換為你的txt檔路徑
    output_file_path = "fake_results.json"  # 替換為輸出的json檔案名稱

    # file_content = read_file(input_file_path)
    # extracted_data = extract_contents(file_content)

    # # 先將所有 question 和 gpt4omini_prompt 寫入 JSON
    # with open(output_file_path, 'w', encoding='utf-8') as json_file:
    #     json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)

    with open(output_file_path, 'r', encoding='utf-8') as json_file:
        data_list = json.load(json_file)

    for data in tqdm(data_list):
        if "gpt4omini_prompt" in data:
            prompt = data["gpt4omini_prompt"]
            response = run_llm(
                prompt=prompt,
                temperature=args.temperature_exploration,
                max_tokens=args.max_length,
                opeani_api_keys=args.opeani_api_keys,
                engine=args.LLM_type
            )
            data[f"{args.LLM_type}_result"] = response

    with open(output_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data_list, json_file, ensure_ascii=False, indent=4)

    print(f"Extraction and LLM processing completed! Results are being saved to {output_file_path}")
