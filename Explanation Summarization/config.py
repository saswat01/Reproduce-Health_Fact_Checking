import torch

lr=5e-5

train_path = ""
test_path = ""
eval_path = ""

model_name = {'bart': 'sshleifer/distilbart-cnn-6-6', 'T5':"t5-small"}

saved_model_path = ""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")