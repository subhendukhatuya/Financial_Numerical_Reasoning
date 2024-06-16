Please follow the below steps to run our codebase.


## Target Computation Module

# Data Folder

DATA Folder: https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link
First unzip "Data_Target_Module" folder in the root folder

# Run the Matching codes:

For FinQA: 
```
python3 Matching_FinQA.py
```

For ConvFinQA: 
```
python3 Matching_ConvFinQA.py
```

# Running the GPT-4 based target module:

Before running these codes please set up Azure endpoints for GPT-4 and paste the API Key in the codes below.

For FinQA: 
```
python3 finqa_run.py
```

For ConvFinQA: 
```
python3 convfinqa_run.py
```

# Retriever Module

These codes are present under the Retriever codes. The outputs of these codes will be used in the Target Computation module. \\ 
The Target Computation Module can be directly run as we provide the outputs from the Retrievers as "Data_Target_Module" folder \\

For running Retriever Codes please refer to Readme in the Retriever Codes folder

# RESULTS

We report all the final files generated under Experiment/Final



