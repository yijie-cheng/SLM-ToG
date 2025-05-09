from openai import OpenAI

def run_llm(prompt, temperature, max_tokens, engine):
    client = OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    engine = client.models.list().data[0].id

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
            print(response)
            result = response.choices[0].message.content
            f = 1
        except Exception as e:
            print(f"Error during request, retrying: {e}")
    
    return result

print(run_llm("hi!", 0, 1024, "meta-llama/Meta-Llama-3-8B-Instruct"))