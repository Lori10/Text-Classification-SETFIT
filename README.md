# Few-shot text classification with SETFIT

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Libraries and Resources](#Libraries-and-Resources)
  * [Data](#Data)
  * [SETFIT model](#SETFIT-model)
  * [Sentence Transformer and Classification Head](#Sentence-Transformer-and-Classification-Head)
  * [Data Generation for finetuning ST](#Data-Generation-for-finetuning-ST)
  * [Advantages of SETFIT](#Advantages-of-SETFIT)
  * [Hard Negative Sampling](#Hard-Negative-Sampling)
  * [Baselines](#Baselines)
  * [Experimental Setup](#Experimental-Setup)
  * [Results](#Results)
  * [Bug and Feature Request](#Bug-and-Feature-Request)

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
SETFIT uses a two-step training methodology, that begins with fine-tuning a Sentence Transformer (ST) and then training a classifier head. In the first step, the ST is fine-tuned to sentence pairs from the input data using a contrastive Siamese approach. This involves encoding the sentence pairs and adjusting the parameters of ST to minimize the distance between semantically similar pairs and maximize the distance between dissimilar pairs. In the subsequent step, a text classification head is trained with the encoded training data generated by the fine-tuned ST from the first step. This classification head learns to classify text based on the encoded representations provided by the ST. These two steps are visually represented in the figure below.
![alt text](https://github.com/Lori10/Text-Classification-SETFIT/blob/main/setfit.png "Image")

## Sentence Transformer and Classification Head
* The main component of SETFIT is a Sentence Transformer. Sentence Transformers are large neural networks designed to optimize and generate high-quality sentence embeddings by relying on pre-trained Transformer-based models such as BERT and RoBERTA. They are fine-tuned on various downstream tasks to learn high-quality sentence representations. Sentence embeddings are vector representations that capture the semantic and contextual information of a sentence. They encode the meaning of a sentence in a fixed-dimensional vector and allow various natural language processing tasks such as text classification, semantic search, and clustering to be performed based on the sentence embeddings. 
* The first of step of applying SETFIT model with our few-shot training data would be to generate new data (triples) from the few-shot training data using a specific approach which is explained in the section 'Data generation for finetuning ST'.
* In the second step, the fine-tuned sentence transformer (ST) processes the original labeled training data denoted as {si}, resulting in a singular sentence representation for each training instance. These representations, accompanied by their respective class labels, comprise the training set for the classification head. A logistic regression model is utilized as the text classification head in case of a binary classification problem.

## Data Generation for finetuning ST
* How do we generate positive and negative samples/triples which are used to finetune the Sentence Transformer?
* Given (s_1, s_2, …, s_K) sentences and (l_1, l_2, …, l_K) labels
* For each s_i randomly sample a “positive” s_j where l_i == l_j and a “negative” s_k where l_i != l_k
* This produces two triples (s_i, s_j, 1) and (s_i, s_k, 0) that are used for contrastive learning. We repeat this process for n iterations.
* SetFit changes the functionality of the ST from sentence embedding to topic embedding.

## Advantages of SETFIT
1. State-of-the-art results: SETFIT achieves state-of-the-art results in few-shot setup which makes its performance comparable with other SOTA techniques like standard fine tuning, in-context learning, parameter-efficient fine tuning and pattern exploiting training.
2. Faster to train: SETFIT is orders of magnitude faster to train or perform inference compared to other SOTA techniques.
3. No need of prompt engineering: Pattern exploiting training and prominent PEFT methods require, as part of their training, the input of manually generated prompts, yielding varying outcomes depending on the level of manual prompt-engineering. On the contrary, SETFIT does not require any prompt design.
4. Run zero-shot classification: Using SETFIT we can also run zero-shot classification (when there is no labeled data available) by producing synthetic data for example: ‘this sentence is negative’ has the label 0 and ‘this sentence is positive‘ has the label 1. This has been proven to result in a very good performance too.

## Baselines
I compare SETFIT’s performance against the following models or approaches:
1. Standard fine tuning of BERT on few-shot training data. We optimize all model’s parameters using batches from a few training examples.
2. Standard Fine Tuning of BERT using the entire training dataset. We optimize all model’s parameters using batches from the entire dataset.
3. SETFIT without fine tuning the ST. Instead of fine tuning the pre-trained sentence embeddings generated by the ST model, we train the classification head based on the pre-trained embeddings.
4. I also compare the performance of random sampling against hard negative/positive sampling.


## Experimental Setup
Since the performance of SETFIT and standard fine-tuning may be sensitive to the choice of few-shot training data, like the original paper I use 5 different random training splits for each dataset and sample size. Similar to the original paper, I use 2 different sample sizes of training data: M=18 and M=50 whereas the original paper uses M=8 and M=64. I use accuracy to evaluate the performance of each method on test dataset since the data is balanced and report the mean like the original paper. I fine-tune SETFIT using cosine-similarity loss, a learning rate of 0.00002, a batch size of 16 and epoch of 1. I also use the same learning, batch size and number of epochs for standard
fine tuning. I set the same values for the hyperparameters across all methods and do not perform any hyperparameter tuning.

## Hard Negative Sampling
My contribution to the original paper of SETFIT is as follows:
1. I propose a new technique for sampling the data in the data generation process which is used for fine-tuning the Sentence Transformer which I called
hard negative/positive sampling. I demonstrate the effectivness of this approach and compare the performance with random sampling.
2. I make the code and data used in my work publicly available.

* Can we improve performance using any approach instead of random sampling?
* We have the following training data: (s_1, s_2, …, s_K) sentences and (l_1, l_2, …, l_K) labels
* For each sentence s_i and for each iteration 1-n, sample a “positive” s_j where l_i == l_j & cos_sim(s_i, s_j) is minimal and a “negative” s_k where l_i != l_k & cos_sim(s_i, s_k) is maximal to produce as a result two triples (s_i, s_j, 1) and (s_i, s_k, 0) whereas cos_sim is a function which represents the cosine similarity.
* Remove s_j and s_k from the sampling list to avoid choosing the same samples in the next iterations. 
* Concatenate all the positive and negative triplets across all class labels to build the data for ST fine-tuning.
* The idea is to choose sentences that are not similar at all but have the same label (make them more similar during fine tuning) and sentences that are very similar but have different labels (make them less similar during fine tuning).
* Expected to work better than random sampling for less nr of iterations (n) and greater nr of training examples (K).

## Results
I find that SETFIT significantly outperforms the standard fine-tuning baseline for M = 18 by an average of 22.6 points(in the original paper by 19.3 points). However, as the number of training samples increases to M = 50, the gap decreases to 13.3 points (in the original paper by 5.6 points). Similarly, standard fine-tuning on entire dataset outperforms SETFIT on average by 5.7 points. For M=18, SETFIT with random sampling has similar performance compared to SETFIT with hard negative sampling. As the size of few-shot training data increases, hard negative/positive sampling outperforms random sampling on average by 1.1 points. SETFIT with fine tuning clearly outperforms SETFIT without fine tuning for M=18 as well as for M=50.

![alt text](https://github.com/Lori10/Text-Classification-SETFIT/blob/main/results.PNG "Image")

## Bug and Feature Request
If you find a bug, kindly open an issue. 
