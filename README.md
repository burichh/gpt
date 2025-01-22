This is my personal project to reproduce the results of OpenAI's GPT2-Small version. I got inspired by Andrej Karpathy's video [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU)

# What is this?
The remarkable power of OpenAI's ChatGPT-4o1 really piqued my curiosity: what's behind all this sorcery? How come that such seemingly simple mathematical operations, matrix multiplications, additions and non-linearity functions chained together can give rise to something that can mimic intelligent human conversation so closely? Is this really what intelligence is? Does it become "alive" just because it can act, very believably so, like a living thing? Fascinating questions, but I am moved utmost by the most fun of all:

### **How does it work?**

What I cannot create, I do not understand!

![](./img/feynman.PNG)

Thus this repo exists.

# Goal
Andrej Karpathy is great. He is just great. When I saw he released his video [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU) my goal was clear: understand every step he makes, the reasoning why he does so, the context within which that reasoning exists, and then reproduce his results. More precisely, reproduce OpenAI's results of GPT2-Small version (124 million parameters), based on the papers [[1]] and [[2]] guided by Andrej's video.

**Learning loop**: watch the video, read the papers (and some others that I've found really helpful too, linked at the bottom), make notes, close the solution, code it from scratch, play with the results, compare with Andrej's code, take over his solutions where I found they were superior, and refactor where needed. Repeat.


# Technical overview

GPT-2 uses the self-attention transformer decoder-only architecture from [[3]], except that the layer normalization was moved to the input of each sub-block and an additional layer normalization was added after the final self-attention block [[1]].

| **Model Parameters**              | #     | Comment                                                                              |
|-----------------------------------|-------|--------------------------------------------------------------------------------------|
| Context length                    | 1024  |                                                                                      |
| Vocabulary size (GPT-2 tokenizer) | 50257 | = 50000 byte pair encoding merges + 256 bytecodes + 1 special <\|endoftext\|> token) |
| Embedding dimension               | 768   |                                                                                      |
| Number of attention heads         | 12    |                                                                                      |
| Head dimension                    | 64    | = Embedding dimension / Number of attention heads                                    |
| Feedforward layer dimension       | 3072  | = 4 * Embedding dimension                                                            |
| Number of transformer layers      | 12    |                                                                                      |

For training I used HuggingFace's [FineWeb Dataset](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1), more specifically the 10 billion token version `sample-10BT` of the [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) which is a collection of high quality educational content scraped from the web.

# Hardware
Originally I planned to run the training on LambdaLabs GPU cloud, on a 8x NVIDIA A100 SXM GPUs (hence I also coded the [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) configuration part, preparing for training on multiple GPUs), but it turned out that my own RTX 4070 Super was sufficient to execute the training with fraction of the price (albeit running for 25x longer).

|         | LambdaLabs On-demand 8x NVIDIA A100 SXM | My own PC NVIDIA RTX 4070 Super                                         |
|---------|-----------------------------------------|-------------------------------------------------------------------------|
| Training time | ~2 hours                                | ~50 hours                                                               |
| Price   | USD 35-40 (including VAT)               | USD 2-3 (based on the current electricity prices and power consumption) |

Thus I picked my own RTX 4070 Super, and went brrr for 2.5 days with it.

# Results
I took over Andrej's logging and plotting code

# References

[1]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf "Language Models are Unsupervised Multitask Learners (GPT-2)"
[[1]] Language Models are Unsupervised Multitask Learners (GPT-2) - [Link](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

[2]: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf "Language Models are Few-Shot Learners"
[[2]] Language Models are Few-Shot Learners (GPT-3) - [Link](https://arxiv.org/pdf/2005.1416)

[3]: https://arxiv.org/pdf/1706.03762 "Attention Is All You Need"
[[3]] Attention Is All You Need - [Link](https://arxiv.org/pdf/1706.03762)



# Why GPT-2 is important?