from tqdm import tqdm
import argparse
import random
from wiki_func import *
from client import *
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="webqsp", help="choose the dataset.")
    parser.add_argument("--max_length", type=int,
                        default=256, help="the max length of LLMs output.")
    parser.add_argument("--temperature_exploration", type=float,
                        default=0.4, help="the temperature in exploration stage.")
    parser.add_argument("--temperature_reasoning", type=float,
                        default=0, help="the temperature in reasoning stage.")
    parser.add_argument("--width", type=int,
                        default=3, help="choose the search width of ToG.")
    parser.add_argument("--depth", type=int,
                        default=3, help="choose the search depth of ToG.")
    parser.add_argument("--remove_unnecessary_rel", type=bool,
                        default=True, help="whether removing unnecessary relations.")
    parser.add_argument("--LLM_type", type=str,
                        default="gpt-3.5-turbo", help="base LLM model.")
    parser.add_argument("--opeani_api_keys", type=str,
                        default="", help="if the LLM_type is gpt-3.5-turbo or gpt-4, you need add your own openai api keys.")
    parser.add_argument("--num_retain_entity", type=int,
                        default=5, help="Number of entities retained during entities search.")
    parser.add_argument("--prune_tools", type=str,
                        default="llm", help="prune tools for ToG, can be llm (same as LLM_type), bm25 or sentencebert.")
    parser.add_argument("--addr_list", type=str,
                        default="server_urls.txt", help="The address of the Wikidata service.")
    args = parser.parse_args()
        
    datas, question_string = prepare_dataset(args.dataset)
    print("Start Running ToG on %s dataset." % args.dataset)
    for data in tqdm(datas[0:200]):
        question = data[question_string]
        warning = {
            "relations_cleaning_error": False,
            "entities_cleaning_error": False,
            "long_context": False,
            "dead_road": False,
            "generate_without_explored_paths": False
        }
        topic_entity = data['qid_topic_entity']

        with open('{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("question: " + question + "\n")
            file.write("topic_entity: " + str(topic_entity) + "\n")
            file.write("------------------------------------------------------------------------------------------------------\n")

        cluster_chain_of_entities = []
        if len(topic_entity) == 0:
            results = generate_without_explored_paths(question, warning, args)
            save_2_jsonl(question, results, [], warning, file_name=args.dataset, LLM_type=args.LLM_type)
            continue
        pre_relations = []
        pre_heads= [-1] * len(topic_entity)
        flag_printed = False
        # server_addrs = ["https://query.wikidata.org/sparql"]
        # print(f"Server addresses: {server_addrs}")
        # wiki_client = MultiServerWikidataQueryClient(server_addrs)
        wiki_client = WikidataQueryClient("dummy_url")
        for depth in range(1, args.depth+1):
            current_entity_relations_list = []
            i=0
            for entity in topic_entity:
                if entity!="[FINISH_ID]":
                    retrieve_relations_with_scores = relation_search_prune(entity, topic_entity[entity], pre_relations, pre_heads[i], question, args, wiki_client, warning)  # best entity triplet, entitiy_id
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i+=1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []

            for entity in current_entity_relations_list:
                value_flag=False
                if entity['head']:
                    entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation'], wiki_client, True)
                else:
                    entity_candidates_id, entity_candidates_name = entity_search(entity['entity'], entity['relation'], wiki_client, False)
                if len(entity_candidates_name)==0:
                    continue
                if len(entity_candidates_id) ==0: # values
                    value_flag=True
                    if len(entity_candidates_name) >=20:
                        entity_candidates_name = random.sample(entity_candidates_name, 10)
                    entity_candidates_id = ["[FINISH_ID]"] * len(entity_candidates_name)
                else: # ids
                    entity_candidates_id, entity_candidates_name = del_all_unknown_entity(entity_candidates_id, entity_candidates_name)
                    if len(entity_candidates_id) >=20:
                        indices = random.sample(range(len(entity_candidates_name)), 10)
                        entity_candidates_id = [entity_candidates_id[i] for i in indices]
                        entity_candidates_name = [entity_candidates_name[i] for i in indices]

                if len(entity_candidates_id) ==0:
                    continue

                scores, entity_candidates, entity_candidates_id = entity_score(question, entity_candidates_id, entity_candidates_name, entity['score'], entity['relation'], args, warning)
                
                total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head = update_history(entity_candidates, entity, scores, entity_candidates_id, total_candidates, total_scores, total_relations, total_entities_id, total_topic_entities, total_head, value_flag)

            if len(total_candidates) ==0:
                half_stop(question, cluster_chain_of_entities, depth, args, warning)
                flag_printed = True
                break
                
            flag, chain_of_entities, entities_id, pre_relations, pre_heads = entity_prune(total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, args, wiki_client)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question, cluster_chain_of_entities, args, warning)
                if stop:
                    print("ToG stoped at depth %d." % depth)
                    save_2_jsonl(question, results, cluster_chain_of_entities, warning, file_name=args.dataset, LLM_type=args.LLM_type)
                    flag_printed = True
                    break
                else:
                    print("depth %d still not find the answer." % depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        half_stop(question, cluster_chain_of_entities, depth, args, warning)
                        flag_printed = True
                    else:
                        # topic_entity = {qid: topic for qid, topic in zip(entities_id, [wiki_client.query_all("qid2label", entity).pop() for entity in entities_id])}
                        topic_entity = {qid: next((entry["label"] for entry in wiki_client.pid2label(entity) if "label" in entry), "Unname_Entity") for qid, entity in zip(entities_id, entities_id)}
                        continue
            else:
                half_stop(question, cluster_chain_of_entities, depth, args, warning)
                flag_printed = True
        
        if not flag_printed:
            results = generate_without_explored_paths(question, warning, args)
            save_2_jsonl(question, results, [], warning, file_name=args.dataset, LLM_type=args.LLM_type)

        with open('{}-{}-ToG-log.txt'.format(args.LLM_type.replace("/", "-"), args.dataset), 'a') as file:
            file.write("\nWarning:\n")
            file.write("relations_cleaning_error: {}\n".format(warning['relations_cleaning_error']))
            file.write("entities_cleaning_error: {}\n".format(warning['entities_cleaning_error']))
            file.write("long_context: {}\n".format(warning['long_context']))
            file.write("dead_road: {}\n".format(warning['dead_road']))
            file.write("generate_without_explored_paths: {}\n".format(warning['generate_without_explored_paths']))
            file.write("\n------------------------------------------------------------------------------------------------------\n")
            file.write("------------------------------------------------------------------------------------------------------\n")
