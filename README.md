# LLM Sparse Autoencoder Embeddings can be used to train NLP classifiers

Full write up is on [medium](https://medium.com/@lewingtonpitsos/llm-sparse-autoencoder-embeddings-can-be-used-to-train-nlp-classifiers-3889c32cef6d). I also copy-pasted the text [here](./notes/write-up.md) 

# Installation

## Mac/Linux

First Install python 3.10 and make sure you're using it.

Next get those pip dependencies

```
python -m venv v # or your favorite way to create a python virtual environment
source v/bin/activate
pip install -r requirements.txt
```

Run the (very, very sparse) tests

```
python -m pytest
```

You also need to create a credentials file called `.credentials.json` and put it in the root of the project structured exactly the same as `.credentials.example.json`. The only key you actually need is `HF_TOKEN` (a huggingface api token). The other keys are there to help keep track of my training files and run experiments on [runpod](runpod.io/console/gpu-cloud)

## Windows

god help you

# Usage

`cli.py` provides a command line interface for using the code. For more details try, but you should ignore everything except what pertains to the `train` and `params` commands.

```python cli.py --help```

The `train` command reads in a list of parameters and performs a training run for each paremeter set. You can see a list of 60 parameters in `experiments/60-classifiers.json`. The most basic usage is:

```python cli.py train experiments/60-classifiers.json```

By default all metrics will be logged to [Weights and Biases](https://wandb.ai). You can stop this with the `skip_wandb` parameter.

The `params` command is a utility option to make it easier to build large lists of parameter sets. It basically just executes the `app/build_params.py` script.

To run the XGBoost code simply `python -m app.boost`. This will create SAE embeddings as well as store the hidden features (form the middle and the end of the model) for the IMDB dataset and then run XGBoost on all 3 sets of features.

# Citations/References

The Sparse Autoencoder code comes from [sae_lens](https://github.com/jbloomAus/SAELens)

Most of the datasets come from the RAFT leaderboard: https://raft.elicit.org/

The other dataset is the IMDB dataset, which is cited as

```
@InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}
```
