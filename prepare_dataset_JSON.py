import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-base", legacy=False)


# Function to load CSV data
def load_csv_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


# Function to prepare the dataset
def prepare_dataset(input_csv_path, output_csv_path):
    input_data = load_csv_data(input_csv_path)
    output_data = load_csv_data(output_csv_path)

    # Debug: Print row counts
    print(f"Number of rows in input CSV: {len(input_data)}")
    print(f"Number of rows in output CSV: {len(output_data)}")

    dataset = []
    min_length = min(len(input_data), len(output_data))

    for i in range(min_length):
        input_text = input_data.iloc[i].to_json()
        output_text = output_data.iloc[i]['output_text']
        dataset.append({"input_text": input_text, "output_text": output_text})

    return dataset


# Function to tokenize the data
def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=1024)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output_text"], truncation=True, padding="max_length", max_length=1024)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


if __name__ == "__main__":
    # Input CSV Path
    input_csv_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/decision_tree.csv"
    # Output CSV Path
    output_csv_path = "/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/split_output.csv"

    # Prepare and tokenize the dataset
    dataset = prepare_dataset(input_csv_path, output_csv_path)
    dataset = Dataset.from_pandas(pd.DataFrame(dataset))
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Print the dataset to verify
    print(f"Dataset: {dataset}")
    print(f"Tokenized dataset: {tokenized_dataset}")

    # Save the tokenized dataset
    tokenized_dataset.save_to_disk("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/cokcok")
