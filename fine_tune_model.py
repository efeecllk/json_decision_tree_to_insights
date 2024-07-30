from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Load the tokenized dataset
tokenized_dataset = load_from_disk("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/cokcok")

# Verify tokenized dataset
print(f"Tokenized dataset: {tokenized_dataset}")

# Use the entire dataset for training if it's too small
train_dataset = tokenized_dataset
test_dataset = tokenized_dataset

# Verify train and test datasets
print(f"Train dataset: {train_dataset}")
print(f"Test dataset: {test_dataset}")

# Print content of the dataset
print("Sample from train dataset:")
print(train_dataset[0])

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",  # Directory to save training results and checkpoints
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Use small batch size to prevent memory issues
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    no_cuda=True,  # Ensure training is done on CPU
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/models/fine_tuned_model")
tokenizer.save_pretrained("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/models/fine_tuned_model")
