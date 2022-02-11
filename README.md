# Risk Prediction with Conversation Graph

This repo provides a reference implementation of **Risk-ConversationGraph** as described in the paper [Risk Prediction with Conversation Graph]():

![Conversation Graph](pics/graph_architecture.png)


Submitted to SIGKDD 2022 for review
--------------------

While the earnings transcript dataset cannot be publicly released due to the data provider's policy, we make our code implementation publicly available.

We hope that our design can benefit researchers and practitioners and shed light on other financial prediction tasks

Note: Researchers can easily obtain the earnings conference call data from Seekingalpha or databases such as Thomson Reuters StreetEvents.

## How to run the code

### Dependencies

Run the following to install a subset of necessary python packages for our code
```sh
pip install -r requirements.txt
```

### Usage

Train and evaluate our models through `main.py`. Here are some vital options:
```sh
main.py:
  --config: Training configuration. Readers can change it to adapt to their specific tasks.
    (default: 'conversation_graph.yml')
  --trainer: Init and run model. In our design, we bound trainer with their specific model.
    (default: 'ConversationGraph')
  --test: Whether to evaluate the model with the best checkpoint.
```

### Configurations for evaluation
After training, turn on --test to test model with best checkpoint. We select the best checkpoint according to the MSE results on the validation dataset.

### Folder structure:
* `assets`: contains `best_checkpoint.pth` and `lda_topic_rep.pkl`. Necessary for computing MSE, MAE, Spearsman's rho and Kendal's tau.
* `data`: contains `data_2015.pkl`, `data_2016.pkl`, `data_2017.pkl`, `data_2018.pkl`.
* `model`: contains `conversation_graph.py`. Our model's file. We use MLPs to construct the network.
* `utils`: We implement attention, contrastive loss, and other functions here. The InfoNCE PyTorch implementation is referred from [Representation Learning with Contrastive Predictive Coding](https://github.com/RElbers/info-nce-pytorch).

### Data structure
We use pre-trained [Bert-based-un-cased](https://huggingface.co/bert-base-uncased) to generate sentence embeddings. Note that other pretrained language models such as [RoBERTa](https://huggingface.co/roberta-base), SentenceBert [Sentence Bert](https://www.sbert.net/), and [FinBert](https://huggingface.co/yiyanghkust/finbert-pretrain) can also be used as the text encoder, but we find the results are similar.


### Train LDA topic model
We select the topic amounts of the LDA model by coherence score:
```python
import os
import gensim
from gensim.models import CoherenceModel
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics
    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics
    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    mallet_path = 'packages/mallet-2.0.8/bin/mallet'
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path,
                                                 corpus=corpus,
                                                 num_topics=num_topics,
                                                 id2word=dictionary,
                                                 random_seed=1234,
                                                 workers=os.cpu_count())
        # model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values
```
Some NLP preprocessing techniques are needed, including converting text to lowercase, removing emojis, expanding contractions, removing punctuation, removing numbers, removing stopwords, lemmatization, etc.
