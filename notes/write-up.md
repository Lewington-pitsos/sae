Below are some experiments I ran using GPT2-Small to classify sentiment on the IMDB Sentiment dataset using pre-trained Sparse Autoencoder Features.


They seem to show a small classifier trained on GPT2 Sparse Autoencoder Features (see golden gate claude for details on SAE’s) out-performing various other straightforward methods for getting GPT2 to act as a classifier. The full results are available on Weights & Biases and can be replicated using this repo.

The upshot is that training Autoencoders (sparse or otherwise) on LLMs may lead to better LLM classification results.

Wait, aren’t LLMs good classifiers already?
Not really.

Large Langauge Models like LLama, GPT-4 etc. seem to be able to parse and understand plaintext to a near human level, but in practice actually getting encoder-only LLMs to act as classifiers turns out to be a pain in the ass.

We could try to go into why, but the bottom line is that essentially all mature classification benchmarks in NLP outside of the few-shot domain (e.g. IMBD Sentiment) are still topped by comparatively tiny and quaint BERT-variants.

This is frustrating because in practice most real-world NLP tasks that people and companies actually need solved eventually boil down to some form of classification, or possibly regression. It’s a shame that we don’t have a good way of leveraging the immense power LLMs to these ends.

So Sparse Autoencoders are the solution?
They might be? This is what a Sparse Autoencoder pre-trained on a LLM looks like:


You train it by having it observe the intermediate activations that flow through a LLM as the LLM processes different sequences. Eventually the Autoencover starts to be able to pull out human interpratable “features” which activate only in response to certain words or phrases being processed by the host LLM, features like “words relating to mechanical devices”, or “landmarks” or “the golden gate bridge”.

It’s highly fascinating. A really good high level explanation by the some of the pioneers is here, technical details can be found here and there are even some in-depth explainer youtube videos like this.

Given this is what a Sparse Autoencoder does the really dumb and obvious idea is to just take those interpratable features, plonk them into classifier (XGBoost, logistic regression, torch.nn.Linear, whatever) and train that classifier to perform some classification task in the usual way.


You can see some experiments in this domain on the Weights & Biases page and the code is available here. Ultimately performance is much worse than SOTA but the technique which worked best was training XGBoost on the features from the gpt2-small-res-jb.blocks.8.hook_resid_pre Sparse Autoencoder from sae_lens.

Ok, so what?
Unfortunately using Sparse Autoencoder features doesn’t really have any practical applications at this stage because all the CHONKY LLMs don’t seem to have high quality Autoencoders trained for them yet, and the results I’m getting with the small LLMS like gpt2-small are still a lot worse than BERT. The upshot is that when someone does build and release them (looking at you Joseph Bloom) we might finally beat BERT.

Hold on, you’re missing the point
Of course the irony here is that people are training Sparse Autoencoders because they want to get a better idea of what is going on inside LLMs. Using those specifically human interpretable features to then perform automated classification is a bit like using a delicate sculpture to try to batter down a door.

We could probably get better classification results by training a regular, non-sparse, non-interpretable Autoencoder in a similar way.

Ablation
Some other things I tried which did not work:

Few-Shot Tasks
You might expect this SAE classification strategy to perform well on few-shot tasks. To that end I collated labeled versions of 3 of the datasets from the RAFT benchmark and tested the SAE classification strategy on these. The results were around both what the RAFT authors achieved with an Adaboost classifier (with no LLM involvement), and what I could achieve using hidden states.

Bigger LLMs
You might expect Autoencoder features extracted from larger LLMs to better represent the underlying text, and therefore lead to better classifications. I tested this briefly by extracting Sparse Autoencoder features from mistral-7b using the mistral-7b-res-wg.blocks.8.hook_resid_pre autoencoder from sae_lens. I found that this did not improve classification performance over the gpt2-small-res-jb.blocks.8.hook_resid_pre Autoencoder.