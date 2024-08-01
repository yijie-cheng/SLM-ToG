import argparse
import json
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="cwq", help="choose the dataset.")
    parser.add_argument("--constraints_refuse", type=bool,
                        default=True, help="LLM may have refuse erorr, enable this option to skip current sample.")
    args = parser.parse_args()

    ground_truth_datas, question_string, CoT_output_datas = prepare_dataset_for_eval(args.dataset, "../CoT/CoT_cwq.json")
    ground_truth_datas, question_string, ToG_output_datas = prepare_dataset_for_eval(args.dataset, "../ToG/ToG_cwq_new.json")

    CoT_num_right = 0
    CoT_num_error = 0
    ToG_num_right = 0
    ToG_num_error = 0
    data_num = len(CoT_output_datas)

    conclusion = []
    for i in range(data_num):
        question = ground_truth_datas[i]['question']
        answers = align(args.dataset, question_string, CoT_output_datas[i], ground_truth_datas)
        CoT_results = CoT_output_datas[i]['cot_result']
        CoT_response = ''
        CoT_Match = False

        CoT_response = CoT_results
        if exact_match(CoT_response, answers):
            CoT_Match = True
            CoT_num_right+=1
        else:
            CoT_num_error+=1

        # if check_string(CoT_results):
        #     CoT_response = clean_results(CoT_results)
        #     if CoT_response=="NULL":
        #         CoT_response = CoT_results
        #     else:
        #         if exact_match(CoT_response, answers):
        #             CoT_Match = True
        #             CoT_num_right+=1
        #         else:
        #             CoT_num_error+=1
        # else:
        #     CoT_response = CoT_results
        #     if args.constraints_refuse and check_string(CoT_response):
        #         continue
        #     if exact_match(CoT_response, answers):
        #         CoT_Match = True
        #         CoT_num_right+=1
        #     else:
        #         CoT_num_error+=1

        answers = align(args.dataset, question_string, ToG_output_datas[i], ground_truth_datas)
        ToG_results = ToG_output_datas[i]['results']
        ToG_response = ''
        ToG_Match = False

        ToG_response = ToG_results
        if exact_match(ToG_response, answers):
            ToG_Match = True
            ToG_num_right+=1
        else:
            ToG_num_error+=1

        # if check_string(ToG_results):
        #     ToG_response = clean_results(ToG_results)
        #     if ToG_response=="NULL":
        #         ToG_response = ToG_results
        #     else:
        #         if exact_match(ToG_response, answers):
        #             ToG_Match = True
        #             ToG_num_right+=1
        #         else:
        #             ToG_num_error+=1
        # else:
        #     ToG_response = ToG_results
        #     if args.constraints_refuse and check_string(ToG_response):
        #         continue
        #     if exact_match(ToG_response, answers):
        #         ToG_Match = True
        #         ToG_num_right+=1
        #     else:
        #         ToG_num_error+=1

        concluded_data = {
        'question': question,
        'answer': answers,
        'CoT_results': CoT_results,
        'CoT_answer': CoT_response,
        'CoT_Match': CoT_Match,
        'ToG_chains': ToG_output_datas[i]['reasoning_chains'],
        'ToG_results': ToG_results,
        'ToG_answer': ToG_response,
        'ToG_Match': ToG_Match,
        }
        conclusion.append(concluded_data)

    with open('conclusion.json', 'w', encoding='utf-8') as f:
        json.dump(conclusion, f, ensure_ascii=False, indent=4)

    print("Exact Match: {}".format(float(CoT_num_right/len(CoT_output_datas))))
    print("right: {}, error: {}".format(CoT_num_right, CoT_num_error))

    # save_result2json(args.dataset, CoT_num_right, CoT_num_error, len(CoT_output_datas), "CoT")

    print("Exact Match: {}".format(float(ToG_num_right/len(ToG_output_datas))))
    print("right: {}, error: {}".format(ToG_num_right, ToG_num_error))

    # save_result2json(args.dataset, ToG_num_right, ToG_num_error, len(ToG_output_datas), "ToG")
    
