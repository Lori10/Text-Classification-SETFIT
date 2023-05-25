# this file contains some helper functions that are needed to build the project.

from datasets import load_dataset
import datasets
from sentence_transformers import SentenceTransformer
import copy

def get_data(df, seed=42, nr_example_per_class=16, get_full_data=False):
  """
    Build the examples for the few shot setup.
  """

  if get_full_data:
    return df['train'].shuffle(seed=seed), df['test'].shuffle(seed=seed)
  else:
    df_1 = df['train'].shuffle(seed=seed).filter(lambda example: example['label'] == 1).select(range(nr_example_per_class))
    df_0 = df['train'].shuffle(seed=seed).filter(lambda example: example['label'] == 0).select(range(nr_example_per_class))
    test_df = df['test'].shuffle(seed=seed)
    return datasets.concatenate_datasets([df_1, df_0]).shuffle(seed=42), test_df
  
def get_embedding(model_name, docs):
  """
  Encode the sentence embedding using a SentenceTransformer model.
  """
  model = SentenceTransformer(model_name)
  return model.encode(docs)

def preprocess_ade_sent(ade_df, sent_df):
  """
    Preprocess the data for the few shot setup.
  """
  shuffled_ade_df = copy.deepcopy(ade_df['test'].shuffle(seed=42))
  ade_df['test'] = shuffled_ade_df.select(range(3000))
  ade_df['validation'] = shuffled_ade_df.select(range(3000, 5879, 1))

  sent_df = load_dataset('SetFit/SentEval-CR')
  shuffled_sent_df = copy.deepcopy (ade_df['train'].shuffle(seed=42))
  sent_df['validation'] = shuffled_sent_df.select(range(600))
  sent_df['train'] = shuffled_sent_df.select(range(600, 3012, 1))

  return ade_df, sent_df






