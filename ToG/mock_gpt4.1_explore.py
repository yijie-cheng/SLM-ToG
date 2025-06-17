import json
import re
import time
from openai import OpenAI
from tqdm import tqdm

engine = "microsoft/Phi-3-mini-4k-instruct"

def log(msg, file_handle):
    # print(msg)
    print(msg, file=file_handle, flush=True)

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine):
    if "gpt" not in engine.lower():
        # print(f"USE OTHER MODEL: {engine}!!!")
        client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        engine = client.models.list().data[0].id
    else:
        client = OpenAI(api_key=opeani_api_keys)

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
            # print("result: "+result)
            return result
            f = 1
        except Exception as e:
            print(f"Error during request, retrying: {e}")
            time.sleep(2)
    if engine == "casperhansen/vicuna-7b-v1.5-awq":
        return result.replace("\\", "") # fix vicuna output bug

TRUE_PROMPT_PREFIX = """Q: What state is home to the university that is represented in sports by George Washington Colonials men's basketball?
A: First, the education institution has a sports team named George Washington Colonials men's basketball in is George Washington University , Second, George Washington University is in Washington D.C. The answer is {Washington, D.C.}.

Q: Who lists Pramatha Chaudhuri as an influence and wrote Jana Gana Mana?
A: First, Bharoto Bhagyo Bidhata wrote Jana Gana Mana. Second, Bharoto Bhagyo Bidhata lists Pramatha Chaudhuri as an influence. The answer is {Bharoto Bhagyo Bidhata}.

Q: Who was the artist nominated for an award for You Drive Me Crazy?
A: First, the artist nominated for an award for You Drive Me Crazy is Britney Spears. The answer is {Jason Allen Alexander}.

Q: What person born in Siegen influenced the work of Vincent Van Gogh?
A: First, Peter Paul Rubens, Claude Monet and etc. influenced the work of Vincent Van Gogh. Second, Peter Paul Rubens born in Siegen. The answer is {Peter Paul Rubens}.

Q: What is the country close to Russia where Mikheil Saakashvii holds a government position?
A: First, China, Norway, Finland, Estonia and Georgia is close to Russia. Second, Mikheil Saakashvii holds a government position at Georgia. The answer is {Georgia}.

Q: What drug did the actor who portrayed the character Urethane Wheels Guy overdosed on?
A: First, Mitchell Lee Hedberg portrayed character Urethane Wheels Guy. Second, Mitchell Lee Hedberg overdose Heroin. The answer is {Heroin}.

Q: Which of JFK's brother held the latest governmental position?
A:"""

separator = '''------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------'''

def main():
    # input_file = "./results/prunetool_results/gpt4.1-gpt4.1-cwq-ToG-log.txt"
    input_file = "./gpt4.1-gpt4.1-cwq-cheatToG-log.txt"
    output_file = "gpt4.1-explore-Phi3-4k-reason-cwq-cheat.json"
    log_file_path = "mock_Phi3-4k_cwq_cheat_log.txt"

    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    raw_datas = content.split(separator)
    results = []

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        for data in tqdm(raw_datas):
            data = data.strip()

            q_match = re.search(r"question:(.*?)topic_entity:", data, re.DOTALL)
            if not q_match:
                continue
            question_string = q_match.group(1).strip()
            log("\nQUESTION_STRING: \n" + question_string, log_file)

            prompt_match = re.findall(r"PROMPT:(.*?)RESULT:", data, re.DOTALL)
            if not prompt_match:
                log("error is here!", log_file)
                continue
            prompt = prompt_match[-1].strip()
            log("\nPROMPT: \n" + prompt, log_file)

            answer_string = run_llm(prompt, 0, 1024, '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', engine)
            log("\nANSWER_STRING: \n" + answer_string, log_file)

            results.append({
                "question": question_string,
                "results": answer_string
            })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"已成功處理 {len(results)} 筆資料，儲存於 {output_file}")

if __name__ == "__main__":
    main()