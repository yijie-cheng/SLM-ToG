import json
import re
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

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
    print(f"Recording relations...")
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in tqdm(data):
        question = item["question"]
        topic_entity = item["topic_entity"]
        relations = extract_relations(txt_file, question, topic_entity)
        item["relations"] = relations

    with open(output_json, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

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
    print(f"Recording {model} results...")
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in tqdm(data):
        question = item["question"]
        topic_entity = item["topic_entity"]
        relations = item.get("relations", [])
        gpt4o_mini_results = extract_results(log_txt, question, topic_entity, relations)
        item["results"][model] = gpt4o_mini_results

    with open(input_json, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)

def calculate_cross_entropy_with_binarizer(relations, ground_truth, predictions):
    # if not predictions:
    #     return "no prediction"

    relations = list(dict.fromkeys(relations))
    lb = LabelBinarizer()
    lb.fit(relations)

    ground_truth_encoded = np.zeros(len(relations))
    pred_encoded = np.zeros(len(relations))
    
    ground_truth_encoded = np.sum(lb.transform(ground_truth), axis=0)
    ground_prob = ground_truth_encoded / np.sum(ground_truth_encoded)

    if not predictions:
        # Assign uniform probability if predictions are empty
        pred_prob = np.full(len(relations), 1 / len(relations))
    else:
        pred_encoded = np.sum(lb.transform(predictions), axis=0)
        pred_prob = pred_encoded / np.sum(pred_encoded)

    cross_entropy = -np.sum(ground_prob * np.log(pred_prob + 1e-15))
    return round(cross_entropy, 4)


def update_json_with_ce(input_json):
    print(f"Calculate CE...")
    with open(input_json, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in tqdm(data):
        relations = item["relations"]
        ground_truth = item["results"].get("gpt-4o-mini", [])
        item["CE"] = {}
        
        for model, predictions in item["results"].items():
            if model != "gpt-4o-mini":
                cross_entropy = calculate_cross_entropy_with_binarizer(relations, ground_truth, predictions)
                item["CE"][model] = cross_entropy

    with open(input_json, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False, indent=4)


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
output_json_path = "./test.json" 
ground_truth_file = "../ToG/results/ori2_results/gpt-4o-mini-cwq-ToG-log.txt"

# Initialize JSON
transform_json(input_json_path, output_json_path, 200)

# Record relations
update_json_with_relations(output_json_path, ground_truth_file, output_json_path)

# Record results
models_and_logs = {
    "gpt-4o-mini": "../ToG/results/ori2_results/gpt-4o-mini-cwq-ToG-log.txt",
    "Qwen2-0.5B": "../ToG/results/ori2_results/Qwen-Qwen2-0.5B-Instruct-cwq-ToG-log.txt",
    "Qwen2-0.5B-json": "../ToG/results/json_results/Qwen-Qwen2-0.5B-Instruct-cwq-ToG-log.txt",
    "Qwen2-1.5B": "../ToG/results/ori2_results/Qwen-Qwen2-1.5B-Instruct-cwq-ToG-log.txt",
    "Qwen2-1.5B-json": "../ToG/results/json_results/Qwen-Qwen2-1.5B-Instruct-cwq-ToG-log.txt",
    "Qwen2-7B": "../ToG/results/ori2_results/Qwen-Qwen2-7B-Instruct-cwq-ToG-log.txt",
    "Qwen2-7B-json": "../ToG/results/json_results/Qwen-Qwen2-7B-Instruct-AWQ-cwq-ToG-log.txt",
}
for model, log_path in models_and_logs.items():
    update_json_with_results(output_json_path, log_path, model)

# Calculate CE
update_json_with_ce(output_json_path)

# Calculate Avg CE
calculate_average_ce(output_json_path)