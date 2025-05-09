from SPARQLWrapper import SPARQLWrapper, JSON
import torch
from utils import *

SPARQLPATH = "http://localhost:8890/sparql"  # depend on your own internal address and port, shown in Freebase folder's readme.md

# pre-defined sparqls
sparql_head_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ns:%s ?relation ?x .\n}"""
sparql_tail_relations = """\nPREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?relation\nWHERE {\n  ?x ?relation ns:%s .\n}"""
sparql_tail_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\nns:%s ns:%s ?tailEntity .\n}""" 
sparql_head_entities_extract = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT ?tailEntity\nWHERE {\n?tailEntity ns:%s ns:%s  .\n}"""
sparql_id = """PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?tailEntity\nWHERE {\n  {\n    ?entity ns:type.object.name ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n  UNION\n  {\n    ?entity <http://www.w3.org/2002/07/owl#sameAs> ?tailEntity .\n    FILTER(?entity = ns:%s)\n  }\n}"""
    
def check_end_word(s):
    words = [" ID", " code", " number", "instance of", "website", "URL", "inception", "image", " rate", " count"]
    return any(s.endswith(word) for word in words)

def abandon_rels(relation):
    if relation == "type.object.type" or relation == "type.object.name" or relation.startswith("common.") or relation.startswith("freebase.") or "sameAs" in relation:
        return True


def execurte_sparql(sparql_query):
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    return results["results"]["bindings"]


def replace_relation_prefix(relations):
    return [relation['relation']['value'].replace("http://rdf.freebase.com/ns/","") for relation in relations]

def replace_entities_prefix(entities):
    return [entity['tailEntity']['value'].replace("http://rdf.freebase.com/ns/","") for entity in entities]


def id2entity_name_or_type(entity_id):
    sparql_query = sparql_id % (entity_id, entity_id)
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    if len(results["results"]["bindings"])==0:
        return "UnName_Entity"
    else:
        return results["results"]["bindings"][0]['tailEntity']['value']
    
from freebase_func import *
from prompt_list import *
import json
import time
import openai
import re
from prompt_list import *
from rank_bm25 import BM25Okapi
from sentence_transformers import util
from sentence_transformers import SentenceTransformer


def clean_relations(json_string, entity_id, head_relations):
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        print("Invalid JSON format")
        return False, "Invalid JSON format"
    
    if "relations" not in data or not isinstance(data["relations"], list):
        print("No relations found in JSON")
        return False, "No relations found in JSON"

    relations = []
    
    for relation_info in data["relations"]:
        relation = relation_info.get("relation", "").strip()
        score = relation_info.get("score")
        
        if not relation or score is None:
            print("Output uncompleted..")
            return False, "Output uncompleted.."
        
        try:
            score = float(score)
        except ValueError:
            print("Invalid score")
            return False, "Invalid score"
        
        is_head = relation in head_relations
        relations.append({"entity": entity_id, "relation": relation, "score": score, "head": is_head})
    
    if not relations:
        print("No relations found")
        return False, "No relations found"
    
    return True, relations


def if_all_zero(topn_scores):
    return all(score == 0 for score in topn_scores)


def clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations):
    relations = []
    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    i=0
    for relation in topn_relations:
        if relation in head_relations:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": True})
        else:
            relations.append({"entity": entity_id, "relation": relation, "score": topn_scores[i], "head": False})
        i+=1
    return True, relations


def construct_relation_prune_prompt(question, entity_name, total_relations, args):
    # print("\nConstruct_relation_prune_prompt:")
    # print(f"question: {question}, \nentity_name: {entity_name}, \ntotal_relations: {total_relations}")
    return extract_relation_prompt % (args.width, args.width) + question + '\nTopic Entity: ' + entity_name + '\nRelations: '+ '; '.join(total_relations) + "\nA: "
        

def construct_entity_score_prompt(question, relation, entity_candidates):
    return score_entity_candidates_prompt.format(question, relation) + "; ".join(entity_candidates) + '\nScore: '


def relation_search_prune(entity_id, entity_name, pre_relations, pre_head, question, warning, args):
    # print(f"\n\nStart relation_search_prune: \nentity_id: {entity_id}, \nentity_name: {entity_name}, \npre_relations: {pre_relations}, \npre_head: {pre_head}, \nquestion: {question}, \nargs: {args}")
    sparql_relations_extract_head = sparql_head_relations % (entity_id)
    head_relations = execurte_sparql(sparql_relations_extract_head)
    head_relations = replace_relation_prefix(head_relations)
    # print(f"Head relations after prefix replacement: {head_relations}")
    
    sparql_relations_extract_tail= sparql_tail_relations % (entity_id)
    tail_relations = execurte_sparql(sparql_relations_extract_tail)
    tail_relations = replace_relation_prefix(tail_relations)
    # print(f"Tail relations after prefix replacement: {tail_relations}")

    if args.remove_unnecessary_rel:
        head_relations = [relation for relation in head_relations if not abandon_rels(relation)]
        tail_relations = [relation for relation in tail_relations if not abandon_rels(relation)]
        # print(f"Head relations after pruning: {head_relations}")
        # print(f"Tail relations after pruning: {tail_relations}")
    
    if pre_head:
        tail_relations = list(set(tail_relations) - set(pre_relations))
        # print(f"Tail relations after removing pre_relations: {tail_relations}")
    else:
        head_relations = list(set(head_relations) - set(pre_relations))
        # print(f"Head relations after removing pre_relations: {head_relations}")

    head_relations = list(set(head_relations))
    tail_relations = list(set(tail_relations))
    total_relations = head_relations+tail_relations
    total_relations.sort()  # make sure the order in prompt is always equal
    # print(f"Total relations after merging and sorting: {total_relations}")
    
    if args.prune_tools == "llm":
        prompt = construct_relation_prune_prompt(question, entity_name, total_relations, args)
        # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        # print(f"Prompt for LLM: \n{prompt}")
        # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, args.dataset, warning)
        # print(f"LLM result: {result}")
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        # print(f"JSON result: {result}")
        # result = re.sub(r'(?<=: )"(.*?)"(?![:,}])', r'\"\1\"', result)
        # print(f"FIXED result: {result}")
        flag, retrieve_relations_with_scores = clean_relations(result, entity_id, head_relations) 
        # print(f"Cleaned relations with LLM: {retrieve_relations_with_scores}, \nflag: {flag}")
        
    elif args.prune_tools == "bm25":
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    elif args.prune_tools == "sentencebert":
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b', device='cpu')
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 
    elif args.prune_tools == "gtr":
        model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")  # 或 gtr-t5-large / gtr-t5-xl 等
        topn_relations, topn_scores = retrieve_top_docs(question, total_relations, model, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations)
    elif args.prune_tools == "e5":
        model = SentenceTransformer("intfloat/e5-base", device="cpu")
        e5_query = "query: " + question
        relation_map = {f"passage: {r}": r for r in total_relations}
        e5_passages = list(relation_map.keys())
        with torch.no_grad():
            topn_passages, topn_scores = retrieve_top_docs(e5_query, e5_passages, model, args.width)
        topn_relations = [relation_map[p] for p in topn_passages]
        del model
        torch.cuda.empty_cache()
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(
            topn_relations, topn_scores, entity_id, head_relations
        )

    else:
        print("Prune tool not found! Use bm25!")
        topn_relations, topn_scores = compute_bm25_similarity(question, total_relations, args.width)
        flag, retrieve_relations_with_scores = clean_relations_bm25_sent(topn_relations, topn_scores, entity_id, head_relations) 



    if flag:
        # print(f"Final retrieved relations with scores: {retrieve_relations_with_scores}")
        return retrieve_relations_with_scores
    else:
        # print("Returning empty list due to format error or too small max_length")
        warning['relations_cleaning_error'] = True
        with open('{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("Relations cleaning failed.\n")
            file.write("------------------------------------------------------------------------------------------------------\n")
        return [] # format error or too small max_length
    
    
def entity_search(entity, relation, head=True):
    if head:
        tail_entities_extract = sparql_tail_entities_extract% (entity, relation)
        entities = execurte_sparql(tail_entities_extract)
    else:
        head_entities_extract = sparql_head_entities_extract% (entity, relation)
        entities = execurte_sparql(head_entities_extract)


    entity_ids = replace_entities_prefix(entities)
    new_entity = [entity for entity in entity_ids if entity.startswith("m.")]
    return new_entity


def entity_score(question, entity_candidates_id, score, relation, warning, args):
    entity_candidates = [id2entity_name_or_type(entity_id) for entity_id in entity_candidates_id]
    if all_unknown_entity(entity_candidates):
        return [1/len(entity_candidates) * score] * len(entity_candidates), entity_candidates, entity_candidates_id
    entity_candidates = del_unknown_entity(entity_candidates)
    if len(entity_candidates) == 1:
        return [score], entity_candidates, entity_candidates_id
    if len(entity_candidates) == 0:
        return [0.0], entity_candidates, entity_candidates_id
    
    # make sure the id and entity are in the same order
    zipped_lists = sorted(zip(entity_candidates, entity_candidates_id))
    entity_candidates, entity_candidates_id = zip(*zipped_lists)
    entity_candidates = list(entity_candidates)
    entity_candidates_id = list(entity_candidates_id)
    if args.prune_tools == "llm":
        prompt = construct_entity_score_prompt(question, relation, entity_candidates)

        result = run_llm(prompt, args.temperature_exploration, args.max_length, args.opeani_api_keys, args.LLM_type, args.dataset, warning)
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            result = json_match.group(0)
        return [float(x) * score for x in clean_scores(result, entity_candidates, warning, args)], entity_candidates, entity_candidates_id

    elif args.prune_tools == "bm25":
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)
    elif args.prune_tools == "sentencebert":
        model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b', device="cpu")
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    elif args.prune_tools == "gtr":
        model = SentenceTransformer("sentence-transformers/gtr-t5-base", device="cpu")  # 或 gtr-t5-large / gtr-t5-xl 等
        topn_entities, topn_scores = retrieve_top_docs(question, entity_candidates, model, args.width)
    elif args.prune_tools == "e5":
        model = SentenceTransformer("intfloat/e5-base", device="cpu")
        e5_query = "query: " + question
        relation_map = {f"passage: {r}": r for r in entity_candidates}
        e5_passages = list(relation_map.keys())
        with torch.no_grad():
            topn_passages, topn_scores = retrieve_top_docs(e5_query, e5_passages, model, args.width)
        topn_entities = [relation_map[p] for p in topn_passages]
        del model
        torch.cuda.empty_cache()
    else:
        print("Prune tool not found! Use bm25!")
        topn_entities, topn_scores = compute_bm25_similarity(question, entity_candidates, args.width)

    if if_all_zero(topn_scores):
        topn_scores = [float(1/len(topn_scores))] * len(topn_scores)
    return [float(x) * score for x in topn_scores], topn_entities, entity_candidates_id

    
def update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head):
    if len(entity_candidates) == 0:
        entity_candidates.append("[FINISH]")
        entity_candidates_id = ["[FINISH_ID]"]
    candidates_relation = [entity['relation']] * len(entity_candidates)
    topic_entities = [entity['entity']] * len(entity_candidates)
    head_num = [entity['head']] * len(entity_candidates)
    total_candidates.extend(entity_candidates)
    total_scores.extend(scores)
    total_relations.extend(candidates_relation)
    total_entities_id.extend(entity_candidates_id)
    total_topic_entities.extend(topic_entities)
    total_head.extend(head_num)
    return total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head


def half_stop(question, cluster_chain_of_entities, depth, warning, args):
    # print("No new knowledge added during search depth %d, stop searching." % depth)
    warning['dead_road'] = True
    with open('{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
        file.write("No new knowledge added during search depth %d, stop searching.\n" % depth)
        file.write("------------------------------------------------------------------------------------------------------\n")
    answer = generate_answer(question, cluster_chain_of_entities, warning, args)
    save_2_jsonl(question, answer, cluster_chain_of_entities, warning, file_name=args.dataset, LLM_type=args.LLM_type)


def generate_answer(question, cluster_chain_of_entities, warning, args): 
    prompt = answer_prompt + question + '\n'
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + 'A: '
    result = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, args.dataset, warning)
    return result


def entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args):
    zipped = list(zip(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores))
    sorted_zipped = sorted(zipped, key=lambda x: x[5], reverse=True)
    sorted_entities_id, sorted_relations, sorted_candidates, sorted_topic_entities, sorted_head, sorted_scores = [x[0] for x in sorted_zipped], [x[1] for x in sorted_zipped], [x[2] for x in sorted_zipped], [x[3] for x in sorted_zipped], [x[4] for x in sorted_zipped], [x[5] for x in sorted_zipped]

    entities_id, relations, candidates, topics, heads, scores = sorted_entities_id[:args.width], sorted_relations[:args.width], sorted_candidates[:args.width], sorted_topic_entities[:args.width], sorted_head[:args.width], sorted_scores[:args.width]
    merged_list = list(zip(entities_id, relations, candidates, topics, heads, scores))
    filtered_list = [(id, rel, ent, top, hea, score) for id, rel, ent, top, hea, score in merged_list if score != 0]
    if len(filtered_list) ==0:
        return False, [], [], [], []
    entities_id, relations, candidates, tops, heads, scores = map(list, zip(*filtered_list))

    tops = [id2entity_name_or_type(entity_id) for entity_id in tops]
    cluster_chain_of_entities = [[(tops[i], relations[i], candidates[i]) for i in range(len(candidates))]]
    return True, cluster_chain_of_entities, entities_id, relations, heads


def reasoning(question, cluster_chain_of_entities, warning, args):
    prompt = prompt_evaluate + question
    chain_prompt = '\n'.join([', '.join([str(x) for x in chain]) for sublist in cluster_chain_of_entities for chain in sublist])
    prompt += "\nKnowledge Triplets: " + chain_prompt + '\nA: '

    response = run_llm(prompt, args.temperature_reasoning, args.max_length, args.opeani_api_keys, args.LLM_type, args.dataset, warning)
    result = extract_answer(response)
    if if_true(result):
        return True, response
    else:
        return False, response
    



