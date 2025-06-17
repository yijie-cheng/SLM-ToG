from prompt_list import *
import json
import time
import openai
from openai import OpenAI
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

import tiktoken
MAX_CONTEXT_LENGTH = 4096
def count_tokens(messages, model_name):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except:
        encoding = tiktoken.get_encoding("cl100k_base")  # fallback
    num_tokens = 0
    for message in messages:
        num_tokens += 4  # 每則訊息的開銷
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
    num_tokens += 2
    return num_tokens

def retrieve_top_docs(query, docs, model, width=3):
    """
    Retrieve the topn most relevant documents for the given query.

    Parameters:
    - query (str): The input query.
    - docs (list of str): The list of documents to search from.
    - model_name (str): The name of the SentenceTransformer model to use.
    - width (int): The number of top documents to return.

    Returns:
    - list of float: A list of scores for the topn documents.
    - list of str: A list of the topn documents.
    """

    query_emb = model.encode(query)
    doc_emb = model.encode(docs)

    scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()

    doc_score_pairs = sorted(list(zip(docs, scores)), key=lambda x: x[1], reverse=True)

    top_docs = [pair[0] for pair in doc_score_pairs[:width]]
    top_scores = [pair[1] for pair in doc_score_pairs[:width]]

    return top_docs, top_scores


def compute_bm25_similarity(query, corpus, width=3):
    """
    Computes the BM25 similarity between a question and a list of relations,
    and returns the topn relations with the highest similarity along with their scores.

    Args:
    - question (str): Input question.
    - relations_list (list): List of relations.
    - width (int): Number of top relations to return.

    Returns:
    - list, list: topn relations with the highest similarity and their respective scores.
    """

    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")

    doc_scores = bm25.get_scores(tokenized_query)

    relations = bm25.get_top_n(tokenized_query, corpus, n=width)
    doc_scores = sorted(doc_scores, reverse=True)[:width]

    return relations, doc_scores


# def clean_relations(json_string, entity_id, head_relations):
#     print("clean relations:")
#     try:
#         data = json.loads(json_string)
#     except json.JSONDecodeError:
#         print("Invalid JSON format")
#         return False, "Invalid JSON format"
    
#     if "relations" not in data or not isinstance(data["relations"], list):
#         print("No relations found in JSON")
#         return False, "No relations found in JSON"

#     relations = []
    
#     for relation_info in data["relations"]:
#         relation = relation_info.get("relation", "").strip()
#         score = relation_info.get("score")
        
#         if not relation or score is None:
#             print("Output uncompleted..")
#             return False, "Output uncompleted.."
        
#         try:
#             score = float(score)
#         except ValueError:
#             print("Invalid score")
#             return False, "Invalid score"
        
#         is_head = relation in head_relations
#         relations.append({"entity": entity_id, "relation": relation, "score": score, "head": is_head})
    
#     if not relations:
#         print("No relations found")
#         return False, "No relations found"
    
#     return True, relations


# def if_all_zero(topn_scores):
#     return all(score == 0 for score in topn_scores)


# def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
#     relations = []
#     if if_all_zero(topn_scores):
#         topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
#     i=0
#     for relation in topn_relations:
#         if relation in head_relations:
#             relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
#         else:
#             relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
#         i+=1
#     return True, relations

def run_llm(prompt, temperature, max_tokens, opeani_api_keys, engine, dataset, warning, args):
    LLM_type = engine
    with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
        file.write("***START RUNING LLM***\n\n")
        file.write("PROMPT: \n" + prompt + "\n")
        file.write("\n")

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

    # 計算 messages token 數量
    prompt_tokens = count_tokens(messages, engine)

    # 若 prompt + max_tokens > 上限，自動調整 max_tokens
    while prompt_tokens + max_tokens > MAX_CONTEXT_LENGTH:
        new_max_tokens = MAX_CONTEXT_LENGTH - prompt_tokens
        if new_max_tokens <= 0:
            raise ValueError(f"Prompt 太長（{prompt_tokens} tokens），無法產生任何回覆，請縮短 prompt。")
        print(f"max_tokens 從 {max_tokens} 自動降為 {new_max_tokens}（context 限制：{MAX_CONTEXT_LENGTH}）")
        max_tokens = max_tokens/2

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
            max_tokens = max_tokens/2
            time.sleep(2)
    if engine == "casperhansen/vicuna-7b-v1.5-awq":
        return result.replace("\\", "") # fix vicuna output bug
    
    with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
        file.write("RESULT: \n" + result + "\n")
        file.write("\n")
        file.write("PROMPT_TOKENS: " + str(response.usage.prompt_tokens) + "\n")
        file.write("COMPLETION_TOKENS: " + str(response.usage.completion_tokens) + "\n")
        file.write("TOTAL_TOKENS: " + str(response.usage.total_tokens) + "\n")
        file.write("------------------------------------------------------------------------------------------------------\n")

    if response.usage.completion_tokens == max_tokens or response.usage.total_tokens == 8192:
        # print("Too long context: The generated response reached the maximum length and may be truncated.")
        warning['long_context'] = True
        with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("Too long context: The generated response reached the maximum length and may be truncated.\n")
            file.write("------------------------------------------------------------------------------------------------------\n")
    return result


def all_unknown_entity(entity_candidates):
    return all(candidate == "UnName_Entity" for candidate in entity_candidates)


def del_unknown_entity(entity_candidates):
    if len(entity_candidates)==1 and entity_candidates[0]=="UnName_Entity":
        return entity_candidates
    entity_candidates = [candidate for candidate in entity_candidates if candidate != "UnName_Entity"]
    return entity_candidates


def clean_scores(json_string, entity_candidates, warning, args):
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        warning['entities_cleaning_error'] = True
        with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("Entities cleaning failed: Invalid JSON format.\n")
            file.write("------------------------------------------------------------------------------------------------------\n")
        return [1 / len(entity_candidates)] * len(entity_candidates)
    
    if "entities" not in data or not isinstance(data["entities"], list):
        warning['entities_cleaning_error'] = True
        with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("Entities cleaning failed: 'entities' field missing or not a list.\n")
            file.write("------------------------------------------------------------------------------------------------------\n")
        return [1 / len(entity_candidates)] * len(entity_candidates)
    
    scores = []
    entity_map = {entity['name']: entity['score'] for entity in data["entities"] if "name" in entity and "score" in entity}
    
    for candidate in entity_candidates:
        score = entity_map.get(candidate, None)
        if score is None:
            warning['entities_cleaning_error'] = True
            with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
                file.write(f"Entities cleaning failed: '{candidate}' missing in JSON output.\n")
                file.write("------------------------------------------------------------------------------------------------------\n")
            return [1 / len(entity_candidates)] * len(entity_candidates)
        scores.append(float(score))
    
    if len(scores) == len(entity_candidates):
        return scores
    else:
        warning['entities_cleaning_error'] = True
        with open('results/paper_results/{}/{}/{}-{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("Entities cleaning failed: Incorrect number of scores.\n")
            file.write("------------------------------------------------------------------------------------------------------\n")
        return [1 / len(entity_candidates)] * len(entity_candidates)


def save_2_jsonl(question, answer, cluster_chain_of_entities, warning, args, file_name, LLM_type):
    dict = {"question":question, "results": answer, "reasoning_chains": cluster_chain_of_entities, "warning":warning}
    with open('results/paper_results/{}/{}/{}-{}-{}-ToG.jsonl'.format(args.LLM_type.replace("/", "-"), args.prune_tools, args.prune_tools, args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
        json_str = json.dumps(dict)
        file.write(json_str + "\n")


def extract_answer(text):
    start_index = text.find("{")
    end_index = text.find("}")
    if start_index != -1 and end_index != -1:
        return text[start_index+1:end_index].strip()
    else:
        return ""


def if_true(prompt):
    if prompt.lower().strip().replace(" ","")=="yes":
        return True
    return False


def generate_without_explored_paths(question, warning, args):
    warning['generate_without_explored_paths']=True
    prompt = cot_prompt + "\n\nQ: " + question + "\nA:"
    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, args.dataset, warning, args)
    return response


def if_finish_list(lst):
    if all(elem == "[FINISH_ID]" for elem in lst):
        return True, []
    else:
        new_lst = [elem for elem in lst if elem != "[FINISH_ID]"]
        return False, new_lst


def prepare_dataset(dataset_name):
    if dataset_name == 'cwq':
        with open('../data/cwq.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_20':
        with open('../data/cwq_20.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'cwq_353':
        with open('../data/cwq_353.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'webqsp':
        with open('../data/WebQSP.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'RawQuestion'
    elif dataset_name == 'grailqa':
        with open('../data/grailqa.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'simpleqa':
        with open('../data/SimpleQA.json',encoding='utf-8') as f:
            datas = json.load(f)    
        question_string = 'question'
    elif dataset_name == 'qald':
        with open('../data/qald_10-en.json',encoding='utf-8') as f:
            datas = json.load(f) 
        question_string = 'question'   
    elif dataset_name == 'webquestions':
        with open('../data/WebQuestions.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'question'
    elif dataset_name == 'trex':
        with open('../data/T-REX.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'zeroshotre':
        with open('../data/Zero_Shot_RE.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'input'    
    elif dataset_name == 'creak':
        with open('../data/creak.json',encoding='utf-8') as f:
            datas = json.load(f)
        question_string = 'sentence'
    else:
        print("dataset not found, you should pick from {cwq, webqsp, grailqa, simpleqa, qald, webquestions, trex, zeroshotre, creak}.")
        exit(-1)
    return datas, question_string