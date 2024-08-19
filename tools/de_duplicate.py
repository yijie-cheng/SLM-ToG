import json
from collections import OrderedDict
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        default="../CoT/cot_cwq.jsonl", help="choose the input file.")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    result = list(OrderedDict((item['question'], item) for item in data).values())

    with open(args.input_file, "w", encoding="utf-8") as file:
        for item in result:
            json.dump(item, file, ensure_ascii=False)
            file.write("\n")
