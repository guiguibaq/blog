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
* **The time for loading memory** (i.e., loading from VRAM to SRAM): [TODO] 
* **The time for compute**: actually computing the forward pass

Let’s double click on the **memory**. There are two main operations:
* Loading all the parameters of the model takes `t_params_loading = N_params * bytes_per_param / memory_bandwidth`
* Loading KV cache: at inference time, tokens are generated one by one, and the whole forward has to be computed at each new token; time complexity of the attention operation grows in $O(token_lenght^2)$, so it takes $O(token_lenght^3)$ over the whole generation - this is a massive cost! Thankfully, most of the attention layer is the same from previous generations, so a cache can be used: it maps a token array to their relevant embeddings prior to the attention layer. This reduces the complexity of one forward to $O(token_lenght)$, or $O(token_lenght^2)$ over a whole generation. Of course, we need to store the KV cache, which has a memory footprint of `token_lenght * bytes_cached_per_token * batch`, so a loading time of `t_cache_loading = token_lenght * bytes_cached_per_token * batch / memory_bandwidth`
* Therefore the overall memory time is `t_memory = (N_params * bytes_per_param + token_lenght * bytes_cached_per_token * batch) / memory_bandwidth`

Now let’s look at the **compute** time:
* Compute time is the number of operations required by a forward pass, multiplied by the time it takes to compute one operation
* The forward pass is a bunch of linear layers, attention layers and non linearities. The non linearities are very fast, and the most of the attention is cached, therefore it’s common practice to only account for the time spent on the linear layers. A linear layer is a matrix multiply: multiplying a data matrix of size `(batch, in)` by a matrix of weights of size `(in, out)` takes `2 * batch * in * out` operations, or `2 * batch * N_params`. 
* Modern LLMs use Mixture of Experts (MoE), where the Feed Foward after the attention is split into parallel networks, and a learnt gate routes each incoming token to its relevant expert(s). Therefore with MOEs a forward pass requires less computation: we should only acount for the "active" parameters (i.e., we do not count the parameters of the experts that are not used).
* Therefore the computation time is `t_compute = batch * N_active_params / FLOPS`.

Finally, let's note that **memory and compute happen concurently** , so the time for one forward pass is the max of both (instead of the sum): `t_forward = max (t_compute, t_memory)

If `t_compute > t_memory`, we say that the system is “compute bound”, if that’s the opposite we say it’s “memory bound”.

Great, we now have all the basics covered, and we understand the two fundamental equations of this podcast! The rest of the discussion is where Dwarkesh and Reiner actually use those two equations to make educated guesses on the parameters of LLMs at big labs.

# 2. Influence of batch size on costs

Let’s double click on the notion of batch at inference:
* At training time a batch is a subset of the training data
* At inference time, we can batch user requests together: we wait for instance 20 ms, get all the user requests at this time, and batch them together.

The more we wait, the larger the batch size we get, and intuitively, the best economies of scale we might get. But is that really true?

Said differently: Gemini has a fast inference mode, which presumably is linked to lower batch size; can we imagine the opposite: a “slow mode" where the price per token drops? Turns out, **costs only reduces with batch size for regimes of low batch size: past a given point, the cost is batch size is flat.**

Let's see why by representing t_forward as of batch_size: 
* `t_params_loading / batch = N_params * bytes_per_param / (memory_bandwidth * batch)` <= this decreases when batch increases
* `t_cache_loading / batch = token_lenght * bytes_cached_per_token / memory_bandwidth` <= this is fixed
* `t_compute / batch = N_active_params / FLOPS` <= this is fixed

[todo: insert graph]

As we see, increasing the batch amortises the loading of the parameters, but it doesn't reduce the cost per token for compute or the KV cache.

While we discuss batch size, can we **estimate the ideal batch size**? In the ideal case, the system is neither memory nor copmute bound: `t compute = t memory`.
Ignoring the KV cache for simplicity, this leads to `batch * N_active / FLOPS = N_total / memory_bandwith`, or `FLOPS / memory_bandwith = batch * N_active / N_total`.
Plugging typical values: `FLOPS / memory_bandwith = 300` and `N_active / N_total = 1 / 8` we get `ideal_batch_size = 24k`.

# 3. GPU parallelisation
There are several types of GPU parallelism worth noting:
* **Data parallelism**: classic sharding - requests are sent to different GPUs (or clusters of GPUs) and executed in parallel. This isn't talked about in the podcast.
* **Tensor parallelism**: the weights of each matrix are split across GPUs, each GPU does a part of the computation, then the outputs are combined in an all to all communication. This isn't talked about either, but is widely used in production.
* **Pipelining**: the neural network is split into sequential chunks, which are executed one after the other. Dwarkesh and Reiner discuss it, and explain why it is avoided when possible.
* **Expert parallelism**: experts are distributed across GPUs.

This podcast focuses on pipelining and expert parallelism.

Pipelining 

It broadly claims that pipelining contributes to solving the memory capacity, but that it should be avoided if possible.

Let’s first explore its effect on latency - pipeline makes latency worse:
* Each step of the pipeline is executed sequentially, which means that in the naive setup the GPUs are waiting idle for other GPUs to solve their part of the computation. At inference, we can solve this by stacking batches in parallel (but it will have an effect on memory, see later). However at training time, we need to wait for the gradients so there is no solution.
* The bandwidth to move from one cluster to the next is slow (8x more than within the same cluster). In the podcast they estimate it to be 10ms (which is huge given that one forward pass is estimated to take 20ms)

Now let’s see to which extent pipelining helps with memory capacity:
* The parameters are split between cluster, so the N params are divided by the number of clusters
* However, the KV cache isn’t: each GPU should store the KV cache for bla-bla-bla

Therefore pipelining seems to be something to avoid if necessary, and Ilya said “pipelining is not wise”

Expert parallelism and GPU cluster architecture 

Experts are distributed across GPUs, which leads to two all-to-all communication. However this communication is one fast bandwidth, so it’s not a bottleneck.

At this point, Reiner explains a bit how NVIDIA GPU clusters are set up: there is a common VRAM (about ~todo Gb) and ~64 GPUs. The VRAM is accessible to all GPU so it can be thought as shared memory. When GPU load data from VRAM, they do it in parallel, so the cluster memory bandwidth really is the GPU bandwidth * the number of GPU. Increasing the number of GPU is actually where most of the gains in memory bandwidth come from - NVIDIA is working hard to increase the number of GPUs per cluster, and next generation should have ~todo

Chinchilla laws, updated for inference

The compute budget to train a model is broadly model size * number of tokens in the training data. If we have a given budget, how should we split it between model size and training data?

Labs used to favour huge model size, but the Chinchilla paper changed everything: it showed empirically that the optimal is obtained at token number = 20 x model parameters.

However, Reiner and Dwarkesh explain we are now in a “new Chinchilla” optimal, to take into account that the compute budget is now split between 3 parts: pre training, post training and inference.

Broadly soaking, whichever model we get from pre training will have to be served at scale, so we’d rather serve a smaller model to reduce inference cost, even if that means training it for longer.

Reiner does a back of the envelope calculation where he assumes that labs spend as much compute in pre training, RL training and inference. This broadly means that the number of tokens in pre training equals the number of tokens at inference. Let’s do some quick maths:

Inference data = 500M/s * 2 months = 2.6 10^15
Training data = 150 T (this number is confidential but Dwarkesh heard this from his connections at Anthropic)

Those two are roughly equal! (Which validate the theory)

And if we compare to Chinchila optimal: 100B * 20 = 2T

So frontier models operate on 100x more training data than Chinchila optimal.

Context length

Gemini has a pricing structure where you pay 50% more after 200k context length.

Cost increases with context length.

Solving for bytes per token we get ~2kB

Would that make sense? Well bytes per token = 2 bytes * 2 (key and query)* Nlayers * d heads * nb heads ~ 2kB

I think Reiner is missing the bytes here, but we’re looking at orders of magnitude anyway.

So it makes sense!

