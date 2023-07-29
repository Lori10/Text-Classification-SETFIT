# Few-shot text classification with SETFIT

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Libraries and Resources](#Libraries-and-Resources)
  * [Data](#Data)
  * [SETFIT model](#SETFIT-model)
  * [Baselines](#Baselines)
  * [Experimental Setup](#Experimental-Setup)
  * [Sentence Transformers](#Sentence-Transformers)
  * [Data Generation for finetuning ST](#Data-Generation-for-finetuning-ST)
  * [Model Building](#Model-Building)
  * [Model Tuning](#Model-Tuning)
  * [Techniques to improve the model performance](#Techniques-to-improve-the-model-performance)
  * [Important Notes](#Important-Notes)
  * [Disadvantages of N gram language model and faced issues](#Disadvantages-of-N-gram-language-model-and-faced-issues)
  * [Demo](#demo)
  * [Bug and Feature Request](#Bug-and-Feature-Request)
  * [Future scope of project](#future-scope)

## Problem Statement
Recent advances in few-shot learning have led to impressive results in scenarios where only a small amount of labeled data is available. Techniques such as standard fine tuning and few-shot learning with large language models have demonstrated their effectiveness. However, these methods are challenging in practice because they rely on manually generated prompts and require large language models with billions of parameters to achieve high accuracy. SETFIT overcomes these limitations. SETFIT is a prompt-free and efficient framework specifically designed for few-shot fine-tuning of Sentence Transformers (ST). The core idea behind SETFIT is to first fine-tune a pre-trained ST on a small set of text pairs in a contrastive Siamese manner. This process leads to a refined model capable of generating comprehensive text embeddings. Subsequently, these embeddings are utilized to train a classification head. My experiments show that SETFIT achieves comparable results to standard fine-tuning on few-shot training data as well as on the entire dataset, while being trained an order of magnitude faster. I also propose my Hard Negative/Positive Sampling approach to generate the data used to
fine-tune the Sentence Transformer, and compare its performance to the Random Sampling currently used to implement SETFIT.

## Libraries and Resources
**Python Version** : 3.9

**Libraries** : setfit, transformers (huggingface)

**References** : https://arxiv.org/pdf/2209.11055.pdf, https://github.com/huggingface/setfit

## Data
I conduct experiments on 3 text classification datasets: SST2, SentEval-CR and Ade-corpus-v2. These datasets used are available on the Hugging Face Hub under the SETFIT organisation. I split the datasets into training and test datasets.


## SETFIT model
SETFIT uses a two-step training methodology, that begins with fine-tuning a Sentence Transformer (ST) and then training a classifier head. In the first step, the ST is fine-tuned to sentence pairs from the input data using a contrastive Siamese approach. This involves encoding the sentence pairs and adjusting the parameters of ST to minimize the distance between semantically similar pairs and maximize the distance between dissimilar pairs. In the subsequent step, a text classification head is trained with the encoded training data generated by the fine-tuned ST from the first step. This classification head learns to classify text based on the encoded representations provided by the ST

## Baselines
I compare SETFIT’s performance against the following models or
approaches:
1. Standard fine tuning of BERT on few-shot training data. We optimize all
model’s parameters using batches from a few training examples.
2. Standard Fine Tuning of BERT using the entire training dataset. We optimize all model’s parameters using batches from the entire dataset.
3. SETFIT without fine tuning the ST. Instead of fine tuning the pre-trained
sentence embeddings generated by the ST model, we train the classification
head based on the pre-trained embeddings.
4. I also compare the performance of random sampling against hard negative/positive
sampling.


## Experimental Setup
Since the performance of SETFIT and standard finetuning may be sensitive to the choice of few-shot training data, like the original paper I use 5 different random training splits for each dataset and sample size. Similar to the original paper, I use 2 different sample sizes of training data: M=18 and M=50 whereas the original paper uses M=8 and M=64. I use accuracy to evaluate the performance of each method on test dataset since the data is balanced and report the mean like the original paper. I fine-tune SETFIT using cosine-similarity loss, a learning rate of 0.00002, a batch size of 16 and epoch of 1. I also use the same learning, batch size and number of epochs for standard
fine tuning. I set the same values for the hyperparameters across all methods and do not perform any hyperparameter tuning.
