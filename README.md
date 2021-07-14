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
Open the config.py([bert-base-uncased](https://github.com/saswat01/Reproduce-Health_Fact_Checking/blob/main/veracity_prediction_bert-base/config.py), [scibert](https://github.com/saswat01/Reproduce-Health_Fact_Checking/blob/main/veracity_prediction_scibert/config.py)) file. Make sure to make an empty folder/directory inside the bert or scibert directory for saving the checkpoints. You should provide this checkpoint path in the config.py file. Also, hyper parameters like learning rate, epochs, batch size have been set to the best results but can be changed to preform experiment. Change the training, validation and test directory path to match your requirements(please use the [processed](https://github.com/saswat01/Reproduce-Health_Fact_Checking/tree/main/data/processed) data we have provided in the repository).

After deciding your parameters and changing path to data directories run the following to train your model:
```setup
python veracity_prediction_bert-base/train.py
```
OR
```setup
python veracity_prediction_scibert/train.py
```
## Metrics for veracity prediction

For displaying metrics run:

BERT
```setup
python veracity_prediction_bert-base/metrics.py
```

Sci-BERT
```setup
python veracity_prediction_scibert/metrics.py
```

## Results
#### BERT RESULTS
BERT model acheives a test accuracy of 63.40% on test dataset with parameters batch_size=10, learning_rate=1e-6 and epochs=4. Our model was trained on Google Colab with Nvidia T8 GPU.

You can have a look at the bert models classification report below {0:False, 1:True, 2:Mixture, 3:Unproven}:


<img src ="https://github.com/saswat01/Reproduce-Health_Fact_Checking/blob/main/veracity_prediction_bert-base/bert_result.png" alt="bert_classification_report">

#### Sci-BERT RESULTS
Sci-Bert model ahceives a test accuracy of 62.59% with the parameters same as the ones used for the BERT model. This model was also trained on Google Colab with Nvidia T8 GPU.

Classification report for Sci-BERT {0:False, 1:True, 2:Mixture, 3:Unproven}:


