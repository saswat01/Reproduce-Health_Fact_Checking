import configparser
import transformers
import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

class inputData(Dataset):
    def __init__(self, df):
        self.data = df
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        config = configparser.ConfigParser()
        config.read('config.ini')
        self.config = config
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #self.data = self.ranking(self.data)
        rows = self.data.iloc[idx]
        claims = (str(rows.claim)).lower()
        evidences = (str(rows.top_k)).lower()
        label = rows.label
        encoding = self.tokenizer.encode_plus(
            claims+evidences,
            add_special_tokens=True,
            max_length= self.config['Model']['MAX_LEN'],
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            )
        #return claims, evidences
        return dict(
            input_ids=encoding["input_ids"].flatten(),
            attention_mask=encoding["attention_mask"].flatten(),
            labels=torch.tensor(label, dtype=torch.long)
        ) 

# We will use the pytroch wrapper for faster results

class veracityModule(pl.LightningDataModule):

  def __init__(self, train, test, val, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train
    self.test_df = test
    self.val_df = val

  def setup(self, stage=None):
    self.train_dataset = inputData(
      self.train_df
    )

    self.test_dataset = inputData(
      self.test_df
    )

    self.val_dataset = inputData(
        self.val_df
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

  def val_dataloader(self):
    return DataLoader(
      self.val_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )