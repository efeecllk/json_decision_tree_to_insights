import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Load the fine-tuned model
model_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/models/fine_tuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)


def generate_insights(json_input):
    input_text = json.dumps(json_input)
    insights = text_generator(input_text, max_length=1024, truncation=True)
    return insights[0]['generated_text']


if __name__ == "__main__":
    # Load your new JSON input
    json_input_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/new_input.json"
    with open(json_input_path, 'r') as f:
        json_input = json.load(f)

    # Generate insights
    insights = generate_insights(json_input)

    # Save the generated insights to a .txt file
    output_insights_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/generated_insights.txt"
    with open(output_insights_path, 'w') as f:
        f.write(insights)

    print(f"Insights have been saved to {output_insights_path}")
