import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import Trainer, TrainingArguments
from datasets import Dataset
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the CSV file
df = pd.read_csv("Data_Ensemble/ConvFinqa_mistral_flan_train.csv")

# Extract columns from the dataframe
questions = df["Question"].tolist()
mistral_predicted = df["Mistral_Predicted"].tolist()
flan_predicted = df["Flan_Predicted"].tolist()
ground_truth_facts = df["Rel_Fact"].tolist()

# Combine the inputs into the required format
combined_inputs = [
    f"Question: {question}\nMistral Prediction: {mistral}\nFlan Prediction: {flan}\nOverall Best Retrieved Facts:"
    for question, mistral, flan in zip(questions, mistral_predicted, flan_predicted)
]

# Initialize the tokenizer
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir = '/NS/ssdecl/work')

# Prepare the dataset
data = {
    'input_text': combined_inputs,
    'target_text': ground_truth_facts
}
dataset = Dataset.from_dict(data)

# Tokenize the data
def tokenize(batch):
    source = tokenizer(batch['input_text'], padding="max_length", truncation=True, max_length = 512)
    target = tokenizer(batch['target_text'], padding="max_length", truncation=True, max_length = 512)
    source["labels"] = target["input_ids"]
    return source

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

# Convert tokenized inputs to tensors and move to device
def prepare_dataset(batch):
    source = {key: torch.tensor(value) for key, value in batch.items()}
    #source = {key: value.to(device) for key, value in source.items()}
    return source

# Apply preparation function to tokenized dataset
tokenized_dataset = tokenized_dataset.map(prepare_dataset)


test_df = pd.read_csv("Data_Ensemble/ConvFinqa_mistral_flan_test.csv")

# Extract columns from the test dataframe
test_questions = test_df["Question"].tolist()
test_mistral_predicted = test_df["Mistral_Predicted"].tolist()
test_flan_predicted = test_df["Flan_Predicted"].tolist()
test_ground_truth = test_df["Rel_Fact"].tolist()

# Combine the test inputs into the required format
test_combined_inputs = [
    f"Question: {question}\nMistral Prediction: {mistral}\nFlan Prediction: {flan}\nOverall Best Retrieved Facts:"
    for question, mistral, flan in zip(test_questions, test_mistral_predicted, test_flan_predicted)
]

# Prepare the dataset
test_data = {
    'input_text': test_combined_inputs,
    'target_text': test_ground_truth
}
test_dataset = Dataset.from_dict(test_data)
# Tokenize the test data
test_tokenized_dataset = test_dataset.map(tokenize, batched=True, remove_columns=["input_text", "target_text"])

# Initialize the model
model = BartForConditionalGeneration.from_pretrained(model_name, cache_dir = '/NS/ssdecl/work')

# Move the model to the appropriate device
# Define training arguments
# Define training arguments
training_args = TrainingArguments(
   output_dir="./results/bart_large_convfinqa",
   num_train_epochs=40,
   learning_rate=1e-5,
   per_device_train_batch_size=8,
   save_steps=680,
   save_total_limit=2,
   logging_dir='./logs',
   logging_steps=680,
   report_to="none",  # Disable wandb integration
   evaluation_strategy="steps",  # Evaluate every `eval_steps` steps
   eval_steps=680,  # Number of steps before evaluating on the validation set
   metric_for_best_model="eval_loss",  # Metric to use for determining the best model
   greater_is_better=False,  # Whether a higher value of the metric is better
   load_best_model_at_end=True  # Load the best checkpoint at the end of training
)


#Initialize the Trainer
trainer = Trainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset,
   eval_dataset=test_tokenized_dataset,
)

#Train the model
trainer.train()