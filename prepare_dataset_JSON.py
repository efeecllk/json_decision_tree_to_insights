import json
import pandas as pd
from transformers import AutoTokenizer

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

#json_file_loc= '/Users/efecelik/Desktop/3274_tree3_fine_tune.json'
#Load JSON
def load_json_data(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def load_output_text(output_text_path):
    with open(output_text_path, 'r') as f:
        output_text = f.read()
    return output_text


#load_json_data(json_file_loc)
#print(load_json_data(json_file_loc))

def prepare_dataset(json_path, output_text_path):
    data= load_json_data(json_path)
    output_text= load_output_text(output_text_path)
    dataset= []
    for item in data:
        input_text = json.dumps(item)
        dataset.append({"input_text": input_text, "output_text": output_text})
    return dataset

def tokenize_function(examples):
    model_inputs = tokenizer(examples["input_text"], truncation=True, padding="max_length", max_length=1024)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["output_text"], padding="max_length", max_length=1024)
    model_inputs['labels'] = labels['input_ids']


if __name__ == "__main__":
    from datasets import Dataset
    # JSON Path
    json_path = "/Users/efecelik/Desktop/3274_tree3_fine_tune.json"
    # Output text
    output_text_path = "/Users/efecelik/Desktop/3274_tree3_fine_tune.txt"

    dataset= prepare_dataset(json_path, output_text_path)
    dataset= Dataset.from_pandas(pd.DataFrame(dataset))
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    tokenized_dataset.save_to_disk("/Users/efecelik/Desktop/tokenized_data")









