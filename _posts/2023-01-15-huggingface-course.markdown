---
layout: post
title: HuggingFace Course
date: 2023-01-15 14:00
comments: true
external-url:
categories: Learn
---

# Transformer models


Mentions : 

All the Transformer models mentioned above (GPT, BERT, BART, T5, etc.) have been trained as _language models_.

`pretrained model`: A pre-trained model is *saved network that was previously trained on a large dataset, typically on a large-scale image-classification task*. You either use the pretrained model as is or use transfer learning to customize this model to a given task.

`transfer learning`: Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.

<!-- more -->

`Fine tuning` : Fine-tuning is *a way of applying or utilizing transfer learning*. Specifically, fine-tuning is a process that takes a model that has already been trained for one given task and then tunes or tweaks the model to make it perform a second similar task.

`causal language modeling` : An example of a task is predicting the next word in a sentence having read the _n_ previous words. This is called _causal language modeling_ because the output depends on the past and present inputs, but not the future ones.

`masked language modeling` : Another example is _masked language modeling_, in which the model predicts a masked word in the sentence.

`Pretraining` is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge.

## General architecture

**Introduction**
The model is primarily composed of two blocks:

- `Encoder (left)`: The encoder receives an input and builds a representation of it (its features). This means that the model is optimized to acquire understanding from the input.  

- `Decoder (right)`: The decoder uses the encoder’s representation (features) along with other inputs to generate a target sequence. This means that the model is optimized for generating outputs.

Each of these parts can be used independently, depending on the task:

- `Encoder-only models`: Good for tasks that require understanding of the input, such as sentence classification and named entity recognition.
- `Decoder-only models`: Good for generative tasks such as text generation.  

- `Encoder-decoder models` or `sequence-to-sequence models`: Good for generative tasks that require an input, such as translation or summarization.

## Attention layers

A key feature of Transformer models is that they are built with special layers called _attention layers_. For now, all you need to know is that *this layer will tell the model to pay specific attention to certain words in the sentence you passed it (and more or less ignore the others) when dealing with the representation of each word.*

## Architectures vs. checkpoints

As we dive into Transformer models in this course, you’ll see mentions of _architectures_ and _checkpoints_ as well as _models_. These terms all have slightly different meanings:

- `Architecture`: This is the skeleton of the model — the definition of each layer and each operation that happens within the model.
- `Checkpoints`: These are the weights that will be loaded in a given architecture.  

- `Model`: This is an umbrella term that isn’t as precise as “architecture” or “checkpoint”: it can mean both. This course will specify _architecture_ or _checkpoint_ when it matters to reduce ambiguity.

For example, BERT is an architecture while bert-base-cased, a set of weights trained by the Google team for the first release of BERT, is a checkpoint. However, one can say “the BERT model” and “the bert-base-cased model.”

## USING TRANSFORMERS

### Going through the model

We can download our pretrained model the same way we did with our tokenizer.🤗Transformers provides an `AutoModel` class which also has a `from_pretrained()` method:

```python
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

In this code snippet, we have downloaded the same checkpoint we used in our pipeline before (it should actually have been cached already) and instantiated a model with it.

This architecture contains only the base Transformer module: given some inputs, it outputs what we’ll call `hidden states`, also known as `features`. For each model input, we’ll retrieve a `high-dimensional vector` representing the `contextual understanding of that input by the Transformer model`.

>`A high-dimensional vector ?`
> 
 The vector output by the Transformer module is usually large. It generally has three dimensions:
> - `Batch size`: The number of sequences processed at a time (2 in our example).  

> - `Sequence length`: The length of the numerical representation of the sequence (16 in our example).  

> - `Hidden size`: The vector dimension of each model input. Hidden size is number of features of the hidden state for RNN. So if you increase hidden size then you compute bigger feature as hidden state output.
> - [Difference between hidden dimension and n_layers in rnn using pytorch](https://stackoverflow.com/questions/63294347/difference-between-hidden-dimension-and-n-layers-in-rnn-using-pytorch)


### Model heads: Making sense out of numbers

The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension. They are usually composed of one or a few linear layers

The output of the Transformer model is sent directly to the model head to be processed.

In this diagram, the model is represented by its embeddings layer and the subsequent layers. *The embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token. The subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences.*

## Postprocessing the output
The values we get as output from our model don’t necessarily make sense by themselves. Let’s take a look:
```python
print(outputs.logits)
```

```python
tensor([[-1.5607,  1.6123],
        [ 4.1692, -3.3464]], grad_fn=<AddmmBackward>)
```

Our model predicted `[-1.5607, 1.6123]` for the first sentence and `[ 4.1692, -3.3464]` for the second one. Those are not probabilities but _logits_, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities, they need to go through a [SoftMax](https://en.wikipedia.org/wiki/Softmax_function) layer (all 🤗 Transformers models output the logits, as the loss function for training will generally fuse the last activation function, such as SoftMax, with the actual loss function, such as cross entropy):

> *Logits :* 
> In context of deep learning the [logits layer](https://www.tensorflow.org/tutorials/estimators/cnn#logits_layer) means the layer that feeds in to softmax (or other such normalization). The output of the softmax are the probabilities for the classification task and its input is logits layer. The logits layer typically produces values from -infinity to +infinity and the softmax layer transforms it to values from 0 to 1.

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```python
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
```

Now we can see that the model predicted `[0.0402, 0.9598]` for the first sentence and `[0.9995, 0.0005]` for the second one. These are recognizable probability scores.

To get the labels corresponding to each position, we can inspect the `id2label` attribute of the model config (more on this in the next section):
```python
model.config.id2label
```

```python
{0: 'NEGATIVE', 1: 'POSITIVE'}
```

Now we can conclude that the model predicted the following:

-   First sentence: NEGATIVE: 0.0402, POSITIVE: 0.9598
-   Second sentence: NEGATIVE: 0.9995, POSITIVE: 0.0005