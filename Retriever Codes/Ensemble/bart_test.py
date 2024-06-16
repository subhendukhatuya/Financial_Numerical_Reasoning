import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import Dataset
import torch

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define model name and checkpoint path
model_name = "facebook/bart-large"  # Ensure the model name is correct
checkpoint_path = "./results/bart_large_convfinqa/checkpoint-1360"

# Read the test data
test_df = pd.read_csv("Data_Ensemble/ConvFinqa_mistral_flan_test.csv")

# Extract columns from the test dataframe
test_questions = test_df["Question"].tolist()  # Only take the first 100 entries
test_mistral_predicted = test_df["Mistral_Predicted"].tolist()
test_flan_predicted = test_df["Flan_Predicted"].tolist()
test_ground_truth = test_df["Rel_Fact"].tolist()

# Initialize the tokenizer
tokenizer = BartTokenizer.from_pretrained(model_name, cache_dir = '/NS/ssdecl/work')

# Set the decoder start token ID in the configuration
#config = T5Config()
#config.decoder_start_token_id = tokenizer.pad_token_id

# Initialize the model
model = BartForConditionalGeneration.from_pretrained(checkpoint_path, cache_dir = '/NS/ssdecl/work')
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.to(device)
model.eval()

# Combine the test inputs into the required format
test_combined_inputs = [
    f"Question: {question}\nMistral Prediction: {mistral}\nFlan Prediction: {flan}\nOverall Best Retrieved Facts:"
    for question, mistral, flan in zip(test_questions, test_mistral_predicted, test_flan_predicted)
]

# Tokenize the inputs
def tokenize_text(text):
    return tokenizer(text, truncation=True, max_length=512, return_tensors="pt").to(device)

# Generate the predictions
generated_facts = []
i = 0
print("Starting Generation", flush = True)
for input_text in test_combined_inputs:
    i+=1
    if (i % 200 == 1):
        print(i, flush = True)
    text_encoding = tokenize_text(input_text)
    generated_ids = model.generate(
        input_ids=text_encoding['input_ids'],
        attention_mask=text_encoding['attention_mask'],
        max_length=150,
        num_beams=4,
        no_repeat_ngram_size=2,
        length_penalty=1.0,
        early_stopping=True
    )
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#     print(generated_text)
    generated_facts.append(generated_text)

# Save the generated results and inputs into a separate DataFrame
results_df = pd.DataFrame({
    "Question": test_questions,
    "Mistral_Predicted": test_mistral_predicted,
    "Flan_Predicted": test_flan_predicted,
    "Generated_Facts": generated_facts
})

# Save the results DataFrame to a CSV file
results_df.to_csv("Outputs/ConvFinqa_test_bart_large_1.csv", index=False)
