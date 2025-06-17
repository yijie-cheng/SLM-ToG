import json
import re
import numpy as np
from sklearn.preprocessing import LabelBinarizer

def transform_json(input_file, output_file, data_num):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    transformed_data = []
    for item in data[:data_num]:
        question = item["question"]
        topic_entities = item["topic_entity"].values() 
        
        for topic_entity in topic_entities:
            transformed_entry = {
                "question": question,
                "topic_entity": topic_entity,
                "relations": [],
                "results": {
                },
                "CE": {
                }
            }
            transformed_data.append(transformed_entry)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(transformed_data, outfile, ensure_ascii=False, indent=4)

    print(f"JSON initialization complete")

def extract_relations(txt_file, question, topic_entity):
    relations = []
    with open(txt_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    pattern = re.compile(
        rf"Q: {re.escape(question)}\s+Topic Entity: {re.escape(topic_entity)}\s+Relations: (.*?)\s+A:",
        re.DOTALL
    )
    match = pattern.search(content)
    if match:
        relations = [relation.strip() for relation in match.group(1).split(';') if relation.strip()]
    return relations

def update_json_with_relations(input_json, txt_file, output_json):
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in data:
        question = item["question"]
        topic_entity = item["topic_entity"]
        relations = extract_relations(txt_file, question, topic_entity)
        item["relations"] = relations

    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Recording relations complete")

def extract_results(log_file, question, topic_entity, relations):
    matched_results = []
    with open(log_file, 'r', encoding='utf-8') as infile:
        content = infile.read()

    pattern = re.compile(
        rf"Q: {re.escape(question)}\s+Topic Entity: {re.escape(topic_entity)}\s+Relations:.*?\s+A:\s+RESULT:\s+(.*?)(?=\s+(Q:|PROMPT_TOKENS:))",
        re.DOTALL
    )
    match = pattern.search(content)
    if match:
        result_block = match.group(1)
        for relation in relations:
            if relation in result_block:
                matched_results.append(relation)
    return matched_results

def update_json_with_results(input_json, log_txt, model):
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in data:
        question = item["question"]
        topic_entity = item["topic_entity"]
        relations = item.get("relations", [])
        gpt41_results = extract_results(log_txt, question, topic_entity, relations)
        item["results"][model] = gpt41_results

    with open(input_json, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)
    
    print(f"Recording {model} results complete")

def calculate_cross_entropy_with_binarizer(relations, ground_truth, predictions):
    if not predictions:
        return "no prediction"
    
    lb = LabelBinarizer()
    lb.fit(relations)
    
    ground_truth_encoded = np.sum(lb.transform(ground_truth), axis=0)
    ground_prob = ground_truth_encoded / np.sum(ground_truth_encoded)\
    
    pred_encoded = np.sum(lb.transform(predictions), axis=0)
    pred_prob = pred_encoded / np.sum(pred_encoded)

    cross_entropy = -np.sum(ground_prob * np.log(pred_prob + 1e-15))
    return round(cross_entropy, 4)


def update_json_with_ce(input_json):
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in data:
        relations = item["relations"]
        ground_truth = item["results"].get("gpt-4.1", [])
        item["CE"] = {}
        
        for model, predictions in item["results"].items():
            if model != "gpt-4.1":
                cross_entropy = calculate_cross_entropy_with_binarizer(relations, ground_truth, predictions)
                item["CE"][model] = cross_entropy

    with open(input_json, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

    print(f"Recording CE complete")


def calculate_average_ce(input_json):
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    model_ce_totals = {}
    model_ce_counts = {}

    for item in data:
        ce_values = item.get("CE", {})
        for model, ce in ce_values.items():
            if model not in model_ce_totals:
                model_ce_totals[model] = 0.0
                model_ce_counts[model] = 0
            if ce != "no prediction":
                model_ce_totals[model] += ce
                model_ce_counts[model] += 1

    average_ce = {}
    for model in model_ce_totals:
        print(model)
        if model_ce_counts[model] > 0:
            average_ce[model] = model_ce_totals[model] / model_ce_counts[model]
        else:
            average_ce[model] = None

    print("\nAverage CE values:")
    for model, avg in average_ce.items():
        if avg is not None:
            print(f"{model}: {avg:.4f}")
        else:
            print(f"{model}: No valid predictions")

    return average_ce

input_json_path = "../data/cwq.json" 
output_json_path = "./compare_relations.json" 
ground_truth_file = "../ToG/results/prunetool_results/gpt4.1-gpt4.1-cwq-ToG-log.txt"

transform_json(input_json_path, output_json_path, 200)
update_json_with_relations(output_json_path, ground_truth_file, output_json_path)
models_and_logs = {
    "gpt-4.1": "../ToG/results/prunetool_results/gpt4.1-gpt4.1-cwq-ToG-log.txt",
    "Qwen2-0.5B": "../ToG/results/prunetool_results/Qwen2-0.5b-Qwen2-0.5b-cwq-ToG-log.txt",
    "Qwen2.5-0.5B": "../ToG/results/prunetool_results/Qwen2.5-0.5b-Qwen2.5-0.5b-cwq-ToG-log.txt",
    "gemma2-2b": "../ToG/results/prunetool_results/gemma2b-gemma2b-cwq-ToG-log.txt",
    "Phi3-mini-4k": "../ToG/results/prunetool_results/Phi3mini4k-Phi3mini4k-cwq-ToG-log.txt",
    "Qwen2-7B": "../ToG/results/prunetool_results/Qwen2-7b-Qwen2-7b-cwq-ToG-log.txt",
    "Llama3-8B": "../ToG/results/prunetool_results/Llama-8b-Llama-8b-cwq-ToG-log.txt",
}
for model, log_path in models_and_logs.items():
    update_json_with_results(output_json_path, log_path, model)

update_json_with_ce(output_json_path)
calculate_average_ce(output_json_path)