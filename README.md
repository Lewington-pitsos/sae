# LLM Sparse Autoencoder Embeddings can be used to train NLP classifiers


## Installation

### Mac/Linux

Get those pip dependencies

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

### Windows

god help you

## 