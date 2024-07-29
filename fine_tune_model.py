from transformers import AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_from_disk
from prepare_dataset_JSON import tokenizer



tokenized_dataset = load_from_disk("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/data/cokcok")

print(f"Tokenized dataset: {tokenized_dataset}")


dataset = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset= dataset["train"]
test_dataset= dataset["test"]

model= AutoModelForSeq2SeqLM.from_pretrained("t5-small")

training_args = TrainingArguments(
    output_dir="./results",  # Directory to save training results and checkpoints
    evaluation_strategy="epoch",  # Corrected the typo here
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
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


model.save_pretrained("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/models/fine_tuned_model")
tokenizer.save_pretrained("/Users/efecelik/PycharmProjects/json_decision_tree_to_insights/models/fine_tuned_model")


