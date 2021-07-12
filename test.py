"""
This function is used to test the results using the fine-tuned models(bert for veracity prediction and bart for text summarization).
Please go to the if __name__=='__main__' section. Uncomment the lines instructed to uncomment and run the file.

You have to provide a main text which should be a paragraph(string type) of at least k sentences 
and a claim sentence/sentences to test the model.

Note: k is defined as 5 by default. The bert model used is bert-base-uncased.
"""

from sys import path
from nltk import data
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
from transformers import BertTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import argparse


paths = {"bart":"", 'bert':""}

class output():

    def __init__(self, dataset=None,  main_text=None, claim=None, k=5, bertmodel=None, bart_model=None, 
            bert_tokenizer=None, bart_tokenizer=None, sentence_model=None):
        if claim == None or main_text== None:
            n = np.random.choice(range(dataset.shape[0]))
            self.main_text = dataset.main_text[n]
            self.claim = dataset.claim[n]
        if bertmodel == None:
            checkpoint_bert = str(input("Bert checkpoint path: "))
            self.bert = bert_model(checkpoint_bert)
            self.bert_tokenizer = BertTokenizer('bert-base-uncased')
        else:
            self.bert = bertmodel
            self.bert_tokenizer = bert_tokenizer
        if bart_model == None:
            self.bart_model = AutoModelForSeq2SeqLM.from_pretrained(paths['bart'])
            self.bart_tokenizer = AutoTokenizer.from_pretrained('sshleifer/distilbart-cnn-6-6')
        else:
            self.bart_model = bart_model
            self.bart_tokenizer = bart_tokenizer
        self.k = k
        self.sentence_model = sentence_model
        

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

    def bert_output(self):
        claim, top_k = self._select_evidence_sentences()
        encoder = self.bert_tokenizer.encode_plus(
            claim+top_k,
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

        return top_k

    def bart_output(self, top_k):
        summarizer = pipeline("summarization", model=self.bart_model, tokenizer=self.bart_tokenizer)
        print("Claim: ", self.claim)
        print("Top",self.k,"sentences: ", top_k)
        print('summarization a/c to model', summarizer(top_k))


def _parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", type=str, default='use dataset', help="claim sentence to perform veracity prediction and generate evidence")
    parser.add_argument("--main_text", type=str, default='use dataset', help="The main text describing the claim")
    parser.add_argument("--top_k_sen", type=str, default="no", help="top_k sentences from main_text related to claim")
    args = parser.parse_args()
    return args

def main():
    args = _parse_args()
    sentence_model = SentenceTransformer('bert-base-nli-mean-tokens')
    if args.claim == 'use dataset':
        df = pd.read_csv(paths['test_data'])
        out = output(dataset=df, sentence_model=sentence_model)
    else:
        out = output(main_text=args.main_text, claim=args.claim, sentence_model=sentence_model)
    top_k = out.bert_output()
    out.bart_output(top_k)

if __name__=="__main__":
    main()