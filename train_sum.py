import torch
from data import Data
from tqdm.auto import tqdm
from datasets import load_metric
from transformers import AdamW, get_scheduler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
  

config = {'lr':5e-5}


class Training:
    def __init__(self, train_data_path, eval_data_path, test_data_path, hf_name = 'sshleifer/distilbart-cnn-6-6'):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
        loader = Data(train_data_path, test_data_path, eval_data_path)
        self.train_dataloader, self.eval_dataloader, self.test_dataloader = loader.get_pt_dataloaders(self.tokenizer)
        self.is_trained = False

    def train(self, n_epochs=3):
        optimizer = AdamW(self.model.parameters(), lr=config['lr'])
        num_training_steps = n_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps = num_training_steps
            )

        progress_bar = tqdm(range(num_training_steps))
        self.model.train()

        for epoch in range(n_epochs):
            for batch in self.train_dataloader:
                batch = {k:v.to(config['device']) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

        self.is_trained = True

    def save_model(self, path):
        if self.is_trained:
            self.model.save_pretrained(path)
        else:
            print("Please train the model fully before saving")

    def load_model(self, path):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(path)

    def eval(self):
        self.model.eval()
        metric = load_metric('rouge')
        for batch in self.test_dataloader:
            batch = {k:v.to(config['device']) for k, v in batch.item()}
            with torch.no_grad():
                outputs = self.model(**batch)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            decoded_labels = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
            decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
            metric.add_batch(predictions=decoded_preds, references=decoded_labels)

        print(metric.compute())

    
def main():
    pass