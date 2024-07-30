import json


# Function to parse the JSON tree and generate insights
def parse_tree(node, depth=0):
    insights = ""
    indent = "  " * depth  # Indentation for better readability
    if "name" in node:
        insights += f"{indent}- {node['name']}\n"
    if "children" in node:
        for child in node["children"]:
            insights += parse_tree(child, depth + 1)
    return insights


# Function to generate insights from the JSON input
def generate_insights(json_input):
    insights = "Summary:\n"
    for item in json_input:
        insights += parse_tree(item)
    return insights


if __name__ == "__main__":
    # Load your new JSON input
    json_input_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/xxx.json.json"
    with open(json_input_path, 'r') as f:
        json_input = json.load(f)

    # Generate insights
    insights = generate_insights(json_input)

    # Save the generated insights to a .txt file
    output_insights_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/generated_insights.txt"
    with open(output_insights_path, 'w') as f:
        f.write(insights)

    print(f"Insights have been saved to {output_insights_path}")
