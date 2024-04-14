import numpy as np
import torch

from datasets import ClassLabel
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


multi_label_names = ['Ortho', 'Morph']
num_labels = len(multi_label_names)
multi_labels = ClassLabel(names=multi_label_names)

mlb = MultiLabelBinarizer()
mlb.fit([multi_labels.names])

model_checkpoint = 'seq_clf/morph_ortho/roberta_custom_weighted'

threshold = 0.5

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
rcw_model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels,
                                                               problem_type='multi_label_classification'
                                                               )


def get_annotation(quote, correction, tokenizer, model):
    inputs = tokenizer(quote, correction, return_tensors='pt')
    logits = model(**inputs).logits
    probas = torch.sigmoid(logits)
    predictions = np.array(torch.where(probas > threshold, 1, 0))
    predictions = predictions.reshape(-1, multi_labels.num_classes)
    predictions = mlb.inverse_transform(predictions)

    return predictions


def is_morph(quote, correction):
    predictions = get_annotation(quote, correction, tokenizer, rcw_model)
    return predictions and 'Morph' in predictions[0]
