# Description: This script parses the decision tree json file and saves it as a csv file.

import json
import pandas as pd


def parse_tree(node, parent= None):
    rows= []
    if 'children' in node:
        for child in node['children']:
            rows.extend(parse_tree(child, node['name']))

    rows.append({'node': node['name'], 'parent': parent})
    return rows


def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


if __name__ == "__main__":
    json_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/3274_tree3_fine_tune.json"
    data = load_json_data(json_path)

    rows = parse_tree(data)
    df = pd.DataFrame(rows)
    df.to_csv("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/decision_tree.csv", index=False)
    print("Decision tree parsed and saved to decision_tree.csv")