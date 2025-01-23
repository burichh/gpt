This is my personal project to reproduce the results of OpenAI's GPT2-Small version. I got inspired by Andrej Karpathy's video [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU)

# What is this?
The remarkable power of OpenAI's ChatGPT-4o1 really piqued my curiosity: what's behind all this sorcery? How come that such seemingly simple mathematical operations, matrix multiplications, additions and non-linearity functions chained together can give rise to something that can mimic intelligent human conversation so closely? Is this really what intelligence is? Does it become "alive" just because it can act, very believably so, like a living thing? Fascinating questions, but I am moved utmost by the most fun of all:

### **How does it work?**

What I cannot create, I do not understand!

![](./img/feynman.PNG)

Thus this repo exists.

# Why GPT-2 is important?

# Goal
Andrej Karpathy is great. He is just great. When I saw he released his video [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU) my goal was clear: understand every step he makes, the reasoning why he does so, the context within which that reasoning exists, and then reproduce his results. More precisely, reproduce OpenAI's results of GPT2-Small version (124 million parameters), based on the papers [[1]] and [[2]] guided by Andrej's video.

**Learning loop**: watch the video, read the papers (and some others that I've found really helpful too, linked at the bottom), make notes, close the solution, code it from scratch, play with the results, compare with Andrej's code, take over his solutions where I found they were superior, and refactor where needed. Repeat.

# Technical overview

In the following I write a short summary of the technical details of the model and training. It's only a cristallized version of my own notes, a reminder! The thorough description can be found in the [Let's reproduce GPT-2 (124M)](https://youtu.be/l8pRSuU81PU) video, and the corresponding [build-nanogpt](https://github.com/karpathy/build-nanogpt) repo.

## Model

- GPT-2 uses the self-attention transformer decoder-only architecture from [[3]], except that the layer normalization was moved to the input of each sub-block and an additional layer normalization was added after the final self-attention block [[1]].
- The head of the languge model uses the token embedding matrix from the start of the transformer to compute the output logits, i.e. the input and output embedding matrices are the same (weight sharing) [[9]]. *Short reason: tokens that are "synonyms", and thus have similar input embeddings, should really have similar probabilities in the end (because they can be used interchangeably), and thus should have similar output embeddings*
- The model uses the accelerated version of attention (flash attention [[4]]).
- The model was compiled via `torch.compile` [[6]] to reduce unnecessary python interpreter calls and fuse cuda kernels where such opportunities are found by PyTorch.

| **Model Parameters**              | #     | Comment                                                                              |
|-----------------------------------|-------|--------------------------------------------------------------------------------------|
| Context length                    | 1024  |                                         |
| Vocabulary size (GPT-2 tokenizer) | 50257 | = 50000 byte pair encoding merges + 256 bytecodes + 1 special <\|endoftext\|> token |
| Embedding dimension               | 768   |                                                                                      |
| Number of attention heads         | 12    |                                                                                      |
| Head dimension                    | 64    | = Embedding dimension / Number of attention heads                                    |
| Feedforward layer dimension       | 3072  | = 4 * Embedding dimension                                                            |
| Number of transformer layers      | 12    |                                                                                      |

## Training
- For training I used HuggingFace's [FineWeb Dataset](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1), more specifically the 10 billion token version `sample-10BT` of the [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) which is a collection of high quality educational content scraped from the web. Since the 10 billion token dataset cannot be loaded all at once, it is separated to chunks (aka shards) of 100 million tokens. A batch for a single training step is loaded from these shards.
- `TensorFloar32` precision is used [[5]] instead of the highest `float32` precision, resulting in significant speedups during the training.
- AdamW optimizer is used instead of Adam for faster convergence (i.e. lower training loss) and better generalization performance (i.e. lower validation/test error) [[7]], [[8]].
- The batch size is taken from the GPT-3 paper [[2]], from the GPT-3 Small architecture, where they used a 0.5 million batch size. Since this does not fit into a single forward-backward-update cycle (not enough GPU memory), the gradient is accumulated by iterating with smaller batches, calling forward-backward, but **without** calling update or zeroing out the gradients. Once we cumulatively passed roughly 0.5 million tokens, the accumulated gradients are avaraged out and used to update the model parameters. Then, the gradients are zeroed out, and the next gradient accumulation starts.
- Learning rate is changed during training: it is increased linearly at the beginning (warmup phase), up to a maximum value, then decreased to its minimum value via cosine decay.

| **Training Parameters**        | #          | Comment                                                                                                                        |
|--------------------------------|------------|--------------------------------------------------------------------------------------------------------------------------------|
| Number of GPUs in the training | 1          | My NVIDIA RTX 4070 Super                                                                                                       |
| Number of training tokens      | 10 billion | = the fineweb-edu-10B dataset                                                                                                  |
| Batch size                     | 524288     | Around 0.5M in GPT-3 paper, but rounded up a bit to contain lots of power of two (e.g. divisible with the context length 1024) |
| Micro batch size               | 8          | This is what reliably fits to the 12GB VRAM in my RTX 4070 Super                                                               |
| Gradient accumulation step     | 64         | = Batch size / (Micro batch size * Context length * Number of GPUs in the training)                                            |
| Max steps                      | 19073      | = Number of training tokens / Batch size ; it's not an integer, so it is floored to 19073                                      |
| Max learning rate              | 6e-4       | Maximum learning rate is reached after a linear warmup from zero.                                                              |
| Min learning rate              | 6e-5       | Minimum learning rate is reached after a cosine decay from maximum learning rate.                                              |
| Leanring rate warmup steps     | 715        | Warmup spans across the first 715 steps of training.                                                                           |


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

[2]: https://arxiv.org/pdf/2005.14165 "Language Models are Few-Shot Learners"
[[2]] Language Models are Few-Shot Learners (GPT-3) - [Link](https://arxiv.org/pdf/2005.14165)

[3]: https://arxiv.org/pdf/1706.03762 "Attention Is All You Need"
[[3]] Attention Is All You Need - [Link](https://arxiv.org/pdf/1706.03762)

[4]: https://arxiv.org/pdf/2205.14135 "Flash Attention"
[[4]] Flash Attention Fast and Memory-Efficient Exact Attention with IO-Awareness - [Link](https://arxiv.org/pdf/2205.14135)

[5]: https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html "torch.set_float32_matmul_precision"
[[5]] `torch.set_float32_matmul_precision("high")` - [Link](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html)

[6]: https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html "torch.compile"
[[6]] `torch.compile` - [Link](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html)

[7]: https://arxiv.org/pdf/1711.05101 "Decoupled Weight Decay Regularization"
[[7]] "Decoupled Weight Decay Regularization" - [Link](https://arxiv.org/pdf/1711.05101)

[8]: https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html "AdamW"
[[8]] `torch.optim.AdamW` - [Link](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)

[9]: https://arxiv.org/pdf/1608.05859 "Using the Output Embedding to Improve Language Models"
[[9]] Using the Output Embedding to Improve Language Models - [Link](https://arxiv.org/pdf/1608.05859)