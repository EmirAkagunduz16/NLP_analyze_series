import spacy
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx 
from ast import literal_eval
from pyvis.network import Network 
from nltk.tokenize import sent_tokenize
import os
import sys
import pathlib

folder_path = pathlib.Path().parent.resolve()

sys.path.append(os.path.join(folder_path, '../'))

from utils import load_subtitles_dataset


class NameEntityRecognizer:
  
  def __init__(self):
    self.nlp_model = self.load_model()


  def load_model(self):
    nlp_model = spacy.load('en-core-web-trf')
    return nlp_model


  def get_ners_inference(self, script):
    script_sentences = sent_tokenize(script)
    
    ner_output = []
    
    for sentence in script_sentences:
      doc = self.nlp_model(sentence)
      ners = set()
      for entity in doc.ents:
        if entity.label_ == 'PERSON':
          full_name = entity.text
          first_name = full_name.split(" ")[0]
          first_name = first_name.strip()
          ners.add(first_name)
      
      ner_output.append(ners)
    
    return ner_output

  
  def get_ners(self, dataset_path, save_path=None):
    if save_path is not None and os.path.exists(save_path):
      df = pd.read_csv(save_path)
      df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
      return df
    
    # load dataset
    df = load_subtitles_dataset(dataset_path)
    
    # run inference
    df['ners'] = df['script'].apply(self.get_ners_inference)
    
    if save_path is not None:
      df.to_csv(save_path, index=False)

    return df