Please follow the below steps to run our codebase.



### Data Folder

**DATA Folder:** https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link

Unzip "Data_Target_Module" folder in the root folder


### Requirements

Install all requirements listed in requirements.txt


## Retriever Module

These codes are present under the folder "Retriever Codes". The outputs of these codes will be used in the Target Computation module. \\ 
The Target Computation Module can be directly run as we provide the outputs from the Retrievers as "_Data_Target_Module_" folder \\

For running Retriever Codes please refer to Readme in the "Retriever Codes" folder

### Run the Matching codes:

For FinQA: 
```
python3 Matching_FinQA.py
```

For ConvFinQA: 
```
python3 Matching_ConvFinQA.py

```
## In Context Example selection Module

These codes are present under the folder "In_Context_Selection".
For running In Context Example selection Codes please refer to Readme in the "In_Context_Selection" folder


For running Retriever Codes please refer to Readme in the Retriever Codes folder


### Running Target Answer Computation Module:

Before running these codes please set up GPT-4 API Key and update that in the code.

For FinQA: 
```
python3 finqa_run.py
```

For ConvFinQA: 
```
python3 convfinqa_run.py
```




## Results

We report all the final files generated under Experiment/Final



