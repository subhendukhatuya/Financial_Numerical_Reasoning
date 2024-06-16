Please follow the below steps to run our codebase.

## Retriever Module

### Data
Data Folder: https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link
First unzip the folder "Data_Retriever" in this folder. Also unzip the folder "Data_Ensemble" under Ensemble folder. 

## For Flan (All codes present under Flan Folder):

For FinQA:
```
python3 lora_flan_large_finqa_rel_fact.py
```
For ConvFinQA: 
```
python3 lora_flan_large_convfinqa_rel_fact_train.py
```

## For Mistral (All codes present under Mistral Folder):

For training: 
```
python3 mistral_train.py
```
For inference: 
```
python3 mistral_inference.py
```

## For Ensemble (All codes present under Ensemble folder):

For training: 
```
python3 bart_train.py
```
For inference: 
```
python3 bart_inference.py
```

