Please follow the below steps to run our codebase.

## Dynamic Context Selection

requirements:
python==3.8.10
huggingface-hub==0.0.12
numpy==1.23.2
openai==0.23.0
pandas==1.4.3
torch==1.12.1+cu113
transformers==4.21.1

### Data
Unzip the folder "Data_Prompt_Dynamic" from https://drive.google.com/drive/folders/1GCYQSEXsXsk_O3rHhx8duZ8xw2EUXNAF?usp=drive_link

### Code to train policy:

Before running these codes please set up Azure endpoints for GPT-4 and paste the API Key in the codes below.

```
pip install -r requirements.txt
```
cd run_gpt3_rl
```
python learn_policy.py --label exp2 --ckpt_root ../checkpoints --shot_number 2 --prompt_format TQ-SA --seed 2 --model_config bert-base-uncased --train_number 40 --cand_number 10 --lr 0.001 --epochs 20 --embedding_size 128 --batch_size 20 --gpu 0
```

### Inference

```
python3 examples_extraction.py \--label exp4 \--ckpt_root ../checkpoints \--model gpt3_rl \--test_split test \--test_number -1 \--shot_number 4 \--seed 2 \--cand_number 50 \--embedding_size 128 \--model_config bert-base-uncased \--ckpt exp4/ckpt_best_reward_2.pt \--gpu 0
```




