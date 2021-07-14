# Reproducibility of Explainable Automated Fact-Checking for Public Health Claims

This repository is the implementation of [Explainable Automated Fact-Checking for Public Health Claims](https://arxiv.org/abs/2010.09926). Also, this is the submission of our work under the ML Reproducibility Challenge 2021 Spring.
The repository reproduces the veracity prediction as mentioned in the paper with BERT(base-uncased) and Sci-BERT models. We acheive accuracy close to that mentioned in the paper.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Train model for veracity prediction

To train the BERT or Sci-BERT models for veracity prediction. Fork and clone this GitHub Repository in your local environment.

```setup
git clone repository
```
Go to the veracity_prediction_bert-base or veracity_prediction_scibert folder
```setup
cd veracity_prediction_bert-base
```
OR
```setup
cd veracity_prediction_scibert
```
Open the config.py([bert-base-uncased](https://github.com/saswat01/Reproduce-Health_Fact_Checking/blob/main/veracity_prediction_bert-base/config.py), [scibert](https://github.com/saswat01/Reproduce-Health_Fact_Checking/blob/main/veracity_prediction_scibert/config.py)) file. Make sure to make an empty folder/directory inside the train directory for saving the checkpoints. You should change the checkpoint path to this directory. Also, hyper parameters like learning rate, epochs, batch size have been set to the best results but can be changed to preform experiment.
