from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

def preprocess(tokenizer ,examples):
        inputs = [doc for doc in examples["top_k"]]
        model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["explanation"], max_length=128, padding='max_length', truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

class Data:
    def __init__(self, train_path, test_path, val_path):
        self.dataset = load_dataset('csv', data_files = {"train": train_path, "test":test_path, "validation":val_path})

    def tokenize(self, tokenizer):
        self.dataset = self.dataset.map(lambda x: preprocess(tokenizer, x), batched=True, load_from_cache=False)
        self.dataset.set_format('torch')

    def get_pt_dataloaders(self, tokenizer, train_batch_size=8, eval_batch_size=8, test_batch_size=8):
        self.tokenize(tokenizer)
        train_dataloader = DataLoader(self.dataset['train'], shuffle=True, batch_size=train_batch_size)
        eval_dataloader = DataLoader(self.dataset['validation'], batch_size=eval_batch_size)
        test_dataloader = DataLoader(self.dataset['test'], batch_size=test_batch_size)
        return train_dataloader, eval_dataloader, test_dataloader


if __name__ == "__main__":
    pass

        


    