"""
This function is used to test the results using the fine-tuned models(bert for veracity prediction and bart for text summarization).
Please go to the if __name__=='__main__' section. Uncomment the lines instructed to uncomment and run the file.

You have to provide a main text which should be a paragraph(string type) of at least k sentences 
and a claim sentence/sentences to test the model.

Note: k is defined as 5 by default. The bert model used is bert-base-uncased.
"""

from sys import path
import numpy as np # linear algebra
import pandas as pd
import random
import transformers
import torch
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
import pytorch_lightning as pl
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from operator import itemgetter
import pandas as pd
import numpy as np
import os
import nltk
import torch.nn.functional as F
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from bert_starter import bert_model
from transformers import BertTokenizer

class output():

    def __init__(self, main_text, claim, k, bert_model, bart_model, bert_tokenizer, bart_tokenizer, sentence_model):
        self.main_text = main_text
        self.claim = claim
        self.k = k
        self.bert_model = bert_model
        self.bart_model = bart_model
        self.sentence_model = sentence_model
        self.bert_tokenizer= bert_tokenizer
        self.bart_tokenizer = bart_tokenizer

    def _select_evidence_sentences(self):
        """Select top k evidence sentences based on sentence transformer model."""
        sentence_transformer_model = self.sentence_model    
        claim = str(self.claim)
        sentences = [claim] + [sentence for sentence in sent_tokenize(str(self.main_text))]
        sentence_embeddings = sentence_transformer_model.encode(sentences)
        claim_embedding = sentence_embeddings[0]
        sentence_embeddings = sentence_embeddings[1:]
        cosine_similarity_emb = {}

        for sentence, embedding in zip(sentences, sentence_embeddings):
            cosine_similarity_emb[sentence] = np.linalg.norm(cosine_similarity(
                [claim_embedding, embedding]))

        top_k = dict(sorted(cosine_similarity_emb.items(), key=itemgetter(1))[:self.k]) 
        return claim, ' '.join(list(top_k.keys())) 

    def show(self):
        claims, top_k = self._select_evidence_sentences()
        encoder = self.bert_tokenizer.encode_plus(
            claims+top_k,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
        )
        outputs = self.bert_model(encoder['input_ids'], encoder['attention_mask'])
        logits = F.softmax(outputs[1], dim=1)
        pred_label = torch.argmax(logits, dim=1)
        class_names = ['false', 'true', 'mixture', 'unproven']
        print("Bert logits: ", logits)
        print('-'*1000)
        print("Predicted label: ", class_names[pred_label.item()])
        print('-'*1000)
        # return logits, pred_label.item()
        summarizer = pipeline("summarization", model=self.bart_model, tokenizer=self.bart_tokenizer, framework='pt')
        print('*'*1000)
        print("Summarization results:")
        print("Claim: ", claims)
        print("Top",self.k,"sentences: ", top_k)
        print(f"Summary a/c to model :{summarizer(top_k)[0]}")

if __name__=="__main__":
    checkpoint_bert = str(input("Bert checkpoint path: "))
    bert = bert_model(checkpoint_bert)
    bert_tokenizer = BertTokenizer('bert-base-uncased')
    #bart_model
    #bart_tokenizer
    k = 5
    sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
    #main_text =  # Provide main text here
    #claim = # Provide claim sentence here

    # Uncomment the lines below to see results
    #results = output(main_text, claim, k, bert_model, bart_model, bert_tokenizer, bart_tokenizer, sentence_model)
    #results.show()