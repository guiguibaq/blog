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

Let's start with **pipelining**.

Pipelining contributes to solving the momeory capacity, but it has a tool on latency.

Let's first explore its effect on **latency**:
* Each step of the pipeline is executed sequentially, which means that in the naive setup the GPUs are waiting idle for other GPUs to solve their part of the computation. At inference, we can solve this by stacking batches in parallel (see graph below) - however as we'll see this has a consequence on memory usage. At training time, because we have to wait on the gradients, unfortunately there is no other solution than having the GPUs idle for some time (see graph below).
* Moving data from one GPU cluster to the next is slow (8x slower than moving within the cluster): Reiner estimates it to take 10ms, which is huge considering that one forward pass is expected to take 20ms.

[TODO: insert graph of pipelining across GPUs]
