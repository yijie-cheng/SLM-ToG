import json
import argparse


def jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as infile:
        with open(json_file, 'w') as outfile:
            json_lines = infile.readlines()
            json_list = [json.loads(line) for line in json_lines]
            json.dump(json_list, outfile, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="../CoT/cot_cwq.jsonl", help="choose the input file.")
    args = parser.parse_args()

    jsonl_to_json(args.input_file, args.input_file[:-1])
