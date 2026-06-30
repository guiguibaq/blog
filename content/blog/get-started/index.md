---
title: Diggest of "The math behind how LLMs are trained and served"
summary: 
date: 2026-07-01

# Featured image
# Place an image named `featured.jpg/png` in this page's folder and customize its options here.
image:
  caption: 'Image credit: [**Unsplash**](https://unsplash.com)'

authors:
  - me
---

In this podcast Dwarkesh and Reiner deduce key parameters of how inference is run at frontier labs (eg batch size, parallelism, size of models) from simple cost analysis.

The episode is very dense in information, and assumes prior knowledge of the inner workings of LLMs.

This post is a digested version with additional context when required.

# 1. Basics: time for generating one token

The **cost** of running an LLM is directly proportional to the **time** it takes to run inference: `cost per token = inference time per token * GPUs cost per time`.

The whole episode is based on two main equations that estimate the time to run inference; this is split between:
* **The time for loading memory** from VRAM (fast memory) to SRAM (where the computation is actually made)
* **The time for compute**

Let’s double click on the **memory**. There are two main operations:
* Loading all the parameters of the model takes `t_params_loading = N_params * bytes_per_param / memory_bandwidth`
* Loading KV cache: at inference time, tokens are generated one by one, and the whole forward has to be computed at each new token; time complexity of the attention operation grows in $O(token_lenght^2)$, so it takes $O(token_lenght^3)$ over the whole generation - this is a massive cost! Thankfully, most of the attention layer is the same from previous generations, so a cache can be used: it maps a token array to their relevant embeddings prior to the attention layer. This reduces the complexity of one forward to $O(token_lenght)$, or $O(token_lenght^2)$ over a whole generation. Of course, we need to store the KV cache, which has a memory footprint of `token_lenght * bytes_cached_per_token * batch`, so a loading time of `t_cache_loading = token_lenght * bytes_cached_per_token * batch / memory_bandwidth`
* Therefore the overall memory time is `t_memory = (N_params * bytes_per_param + token_lenght * bytes_cached_per_token * batch) / memory_bandwidth`

Now let’s look at the **compute** time:
* Compute time is the number of operations required by a forward pass, multiplied by the time it takes to compute one operation
* The forward pass is a bunch of linear layers, attention layers and non linearities. The non linearities are very fast, and the most of the attention is cached, therefore it’s common practice to only account for the time spent on the linear layers. A linear layer is a matrix multiply: multiplying a data matrix of size `(batch, in)` by a matrix of weights of size `(in, out)` takes `2 * batch * in * out` operations, or `2 * batch * N_params`. 
* Modern LLMs use Mixture of Experts (MoE), where the Feed Foward after the attention is split into parallel networks, and a learnt gate routes each incoming token to its relevant expert(s). Therefore with MOEs a forward pass requires less computation: we should only acount for the "active" parameters (i.e., we do not count the parameters of the experts that are not used).
* Therefore the computation time is `t_compute = batch * N_active_params / FLOPS`.

Finally, let's note that **memory and compute happen concurently** , so the time for one forward pass is the max of both (instead of the sum): `t_forward = max (t_compute, t_memory)`

If `t_compute > t_memory`, we say that the system is “compute bound”, if that’s the opposite we say it’s “memory bound”.

Great, we now have all the basics covered, and we understand the two fundamental equations of this podcast! The rest of the discussion is where Dwarkesh and Reiner  use those two equations to make educated guesses on the parameters of LLMs at big labs.

# 2. Influence of batch size on costs

What does Reiner even mean by "batch size" at inference time?
* At training time a batch is obviously a subset of the training data
* At inference time, user requests are clubbed together in a batch: we wait for instance 20 ms, get all the user requests at this time, and batch them together. Note that we don't quite batch the "user requests" by the "token requests": maybe one answer will take only 100 tokens, while another will take 10k tokens.

The more we wait, the larger the batch size we get, and one might think, the best economies of scale we might get. But is that really true? Turns out, it's not that simple: **cost per token only reduces with batch size for regimes of low batch size: past a given point, the cost per token is flat, and increasing batch size doesn't birng any gain.**

Let's see why looking at how t_forward changes when batch_size increases: 
* `t_params_loading / batch = N_params * bytes_per_param / (memory_bandwidth * batch)` <= this decreases when batch increases
* `t_cache_loading / batch = token_lenght * bytes_cached_per_token / memory_bandwidth` <= this is fixed
* `t_compute / batch = N_active_params / FLOPS` <= this is fixed

So first the cost per token decrease when we increase the batch size, then it becomes stable - at this point, increasing batch size just means that the end user are waiting longer for each batch to start-. The ideal batch size is at the point where cost per token stabilises - in my [simulations](https://colab.research.google.com/drive/1VRZ6jYPpkdm9Nn0tSF24ewvKL8RjzfPs) I find it to be around 5k, which isn't far to Reiner's whiteboard calculations in the Podcast [todo: check by listening the podcast]. At this ideal point, the forward time per token is 1e-7s or 0.5ms per token [todo: check if that's correct].

<img width="1012" height="547" alt="image" src="https://github.com/user-attachments/assets/9b2cb1f4-bd63-402b-9dd0-862382544b92" />


# 3. GPU parallelisation
There are several types of GPU parallelism worth noting:
* **Data parallelism**: classic sharding - requests are sent to different GPUs (or clusters of GPUs) and executed in parallel. This isn't talked about in the podcast.
* **Tensor parallelism**: the weights of each matrix are split across GPUs, each GPU does a part of the computation, then the outputs are combined in an all to all communication. This isn't talked about either, but is widely used in production.
* **Pipelining**: the neural network is split into sequential chunks, which are executed one after the other. Dwarkesh and Reiner discuss it, and explain why it is avoided when possible.
* **Expert parallelism**: experts are distributed across GPUs.

## 3.1. Pipelining

If we do naive pipelining, where we wait for each batch to be complete, most of the GPUs are waiting idle for the next batch, which is very wastefull.


<img width="687" height="277" alt="image" src="https://github.com/user-attachments/assets/c944c659-5c8c-469e-af37-5c0c93b277e8" />

Instead, we can ladder the batches can be laddered, which make the GPUs work continuously:

<img width="670" height="280" alt="image" src="https://github.com/user-attachments/assets/d2284968-0607-4874-ac99-384e2a20d6a8" />

Note that the laddering isn't possible at training time, since each GPU has to wait for the subsequent one to send back the gradients:

<img width="628" height="226" alt="image" src="https://github.com/user-attachments/assets/239f5a82-9bed-4513-9e44-bad59febf994" />

Effect of pipelining on time and capacity:
* **The compute time is worse**, because the bandwidth to transfer the batch from one GPU cluster to another is slow (8x slower than within a cluster): Reiner estimates that is takes 10ms to transfer one batch from a cluster to the next, which is huge considering that one forward pass is expected to take 20ms.
* **The capacity (and memory time) for parameter loading is better**, because each GPU cluster is in charge of less parameters
* **The capacity (and memory time) for KV cache loading is the same**, because each GPU cluster is in charge of the same number of inference examples at a given time

## 3.2. Architecture of GPU clusters

Reiner explains how GPU clusters from NVIDIA are set-up: their flagship [NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) contain 72 GPUs interconnected to about 20TB of VRAM. The VRAM is accessible to all GPUs, so it can be though as shared memory. When GPU load data from VRAM, they do it in parrallel, so the memory bandwidth at the cluster level is the memory bandwidth per GPU * the number of GPUs.

# 4. Chinchilla laws, updated for inference
The compute budger to train a model is broadly `model_size * number_of_tokens_in_training_data`.

Scaling laws answer the question: given a fixed budget, what is the optimal split between model size and number of tokens (e.g., should we prefer training a huge model and less data, or training a small model on larger data?). Frontier labs used to favour model size, but the [Chinchilla paper](https://arxiv.org/abs/2203.15556) changed this practice: is showed empirically that there is a consistent optimal, at `model size = 20 * number of tokens`.

However, Reiner and Dwarkesh explain that we are now in a **"new Chinchilla optimal"**, because frontier labs split their compute budget into three parts: pre-training, post-training and inference. Broadly speaking, whichever model size is chosen, this model will have to be served, and serving a large model is more expensive than serving a small model - therefore **labs prefer using smaller models trained for longer**.

To demonstrate this, Reiner does a back of the envelope calculation, which he assumes that labs spend as much compute in pre-training, post-training, and serving. Because compute scales linearly with number of token, this is equivalent to assuming that labs use roughlty the same number of tokens to train their models, and to serve them.

Assuming that a frontier lab generates 500 million tokens per second for their users, and that each model is available 2 months before being replaced by the next generagion, we get that the number of tokens for inference is `50M/sec * 2 months = 2.6 * 1e14`. Dwarkesh estimates that Anthropic models are trained on 150 trillion (1.5 * 1e14) tokens. As expected, those two are roughly equal!

The models at frontier labs probably have 100 billion parameters - Chinchilla optimal laws would recommend them to be trained on 20*100B = 2T tokens. But as we just saw they are trained on ~150T tokens, so about **100x more training data than Chinchila optimal!**

# 5. Context length

Gemini has a pricing structure where you pay 50% more after 200k context length; why is that?

Reiner show that, past a certain point, the cost per token increases linearly with the context lenght:
* `t_compute = batch * N_active_parameters / FLOPS` <= this does not depend on context length
* `t_params_loading = N_params * bytes_per_param / memory_bandwidth` <= this does not depend on context length
* `t_cache_loading = token_length * bytes_cached_per_token * batch / memory_bandwidth` <= this increases with token_length

[TODO: insert graph]

If we assume that the limit of 200k context length from Gemini corresponds to the point where the time starts to grow linearly with context length, then we can estimate the btyes per token of Gemini's KV cache. Let's ignore t_params_loading for simplicity:
`t_cache_loading > t_compute` <=> `bytes_caches_per_token > N_active_params * memory_bandwidth / (FLOPS * token_length)`
with `memory_bandwidth / FLOPS = 1 / 300`, `N_active_params = 100B`, and `token_length = 200k`, we get `bytes_cached_per_token = 1.6kB`.

Let's check that result through computing the size of the KV cache per token: for each layer (typically `N_layer ~ 80`), for each attention head (typically `n_head ~ 8`), for both the key and value of the attention (multiply by 2), we need to save an embedding (typically `d_head ~ 128`), with a typical precision of float16 (~2 bytes). Therefore we have `bytes_cached_per_token = N_layer * n)heads * d_heads * 2 * 2 ~ 327kB / token`.

[Note: Reiner doesn't multiply by the number of layers, nor by the 2 bytes, which is why his result is 100x lower than mine. Either he is missing something, or I am, not quite sure].


