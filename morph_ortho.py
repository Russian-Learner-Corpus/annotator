import numpy as np
import torch
from transformers import *
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from datasets import ClassLabel
from sklearn.preprocessing import MultiLabelBinarizer


# multi_label_names = ['Ortho', 'Morph']
# num_labels = len(multi_label_names)
# multi_labels = ClassLabel(names=multi_label_names)
#
# mlb = MultiLabelBinarizer()
# mlb.fit([multi_labels.names])

# model_checkpoint = 'seq_clf/morph_ortho/roberta_custom_weighted'


model = BertForSequenceClassification.from_pretrained("DeepPavlov/rubert-base-cased", num_labels=1)
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased', do_lower_case=True)
model = torch.load('deep_pavlov_rubert_orpho_replace', map_location=torch.device('cpu'))
max_length = 200


def prep_text_for_model(original, corrected, tokenizer, max_length=200):
  input = []
  for num, s in enumerate(original):
    input.append(s+' | '+corrected[num])
  test_encodings = tokenizer.batch_encode_plus(input,max_length=max_length,pad_to_max_length=True)
  test_input_ids = test_encodings['input_ids']
  test_token_type_ids = test_encodings['token_type_ids']
  test_attention_masks = test_encodings['attention_mask']
  # Make tensors out of data
  test_inputs = torch.tensor(test_input_ids)
  # test_labels = torch.tensor(test_labels)
  test_masks = torch.tensor(test_attention_masks)
  test_token_types = torch.tensor(test_token_type_ids)
  # Create test dataloader
  test_data = TensorDataset(test_inputs, test_masks, test_token_types)
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)
  # Save test dataloader
  # torch.save(test_dataloader,'test_data_loader')
  return test_data, test_dataloader


def ortho_morph_predict(original, corrected, tokenizer=tokenizer, model=model, threshold=0.4):
  """Returns model prediction if mistake belongs to morth or ortho class

  Parameters
  ----------
  original : list
      list of original texts with mistakes in it
  corrected : list
      list of corrected texts.
      NB Length and order of the list must match the list in the original variable
  model : pytorch model
        Any pytorch model that will predict mistake class
  tokenizer : tokenizer
        tokenizer from model
  threshold : threshold that devides Ortho(below) and Morth(above)

  Returns
  ------
  List of predicted mistake type (ortho or morph) for each item in the list
  """
  # Prepearing text for model

  test_td, test_dataloader = prep_text_for_model(original, corrected, tokenizer)

  # Track variables
  logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

  # Predict
  for i, batch in enumerate(test_dataloader):
    batch = tuple(t.to('cpu') for t in batch)
    # Unpack the inputs from our dataloader
    b_input_ids, b_input_mask, b_token_types = batch
    with torch.no_grad():
      # Forward pass
      outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
      b_logit_pred = outs[0]
      pred_label = torch.sigmoid(b_logit_pred)

      b_logit_pred = b_logit_pred.detach().cpu().numpy()
      pred_label = pred_label.to('cpu').numpy()

    tokenized_texts.append(b_input_ids)
    logit_preds.append(b_logit_pred)
    pred_labels.append(pred_label)

  tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
  pred_labels = [item for sublist in pred_labels for item in sublist]
  # Converting flattened binary values to boolean values
  print(pred_labels)
  pred_labels = [True if pl>threshold else False for pl in pred_labels]
  return pred_labels
