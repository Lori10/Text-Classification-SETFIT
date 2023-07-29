# Few-shot text classification with SETFIT

## Table of Content
  * [Problem Statement](#Problem-Statement)
  * [Data](#Data)
  * [Used Libraries and Resources](#Used-Libraries-and-Resources)
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
Recent advances in few-shot learning have led to impressive results in scenarios where only a small amount of labeled data is available. Techniques such as standard fine tuning and few-shot learning with large language models have demonstrated their effectiveness. However, these methods are challenging in practice because they rely on manually generated prompts and require large language models with billions of parameters to achieve high accuracy. SETFIT overcomes these limitations. SETFIT is a prompt-free and efficient framework specifically designed for few-shot fine-tuning of Sentence Transformers (ST). The core idea behind SETFIT is to first fine-tune a pre-trained ST on a small set of text pairs in a contrastive Siamese manner. This process leads to a refined model capable of generating comprehensive text embeddings. Subsequently, these embeddings are utilized to train a classification head. My experiments show that SETFIT achieves comparable results to standard fine-tuning on few-shot training data as well as on the entire dataset, while being trained an order of magnitude faster. I also propose my Hard Negative/Positive Sampling approach to generate the data used to
fine-tune the Sentence Transformer, and compare its performance to the Random Sampling currently used to implement SETFIT.
