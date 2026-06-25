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

Basics: time for generating one token

The whole episode is based on two key equations that estimate how much time is required to compute one token at inference.

There are two equations because to run inference the GPU needs to do two things:
* Memory: load the parameters it needs of the models from VRAM to SRAM. todo
* Compute: actually compute the forward pass 

Let’s double click on the memory:
* Loading parameters of the model from VRAM to SRAM: todo
* Loading KV cache: to generate a sequence (the, cat, sits, on, the), we generate one token at a time, so (the), (the, cat) etc. That creates a lot of repetition. To avoid this, it is cached: key=array of token, values=arr of v. Todo

Now let’s look at the compute time:
* Compute time is the number of operations required by a forward pass, multiplied by the time it takes to compute one operation
* The forward pass is a bunch of linear layers, attention layers and non linearities. The non linearities are very fast, and the most of the attention is cached, therefore it’s common practice to only count time spent on the linear layers. A linear layer is a matrix multiply: multiplying a data matrix of size (B, in) by a matrix of weights of size (in, out) takes 2 * B * in * out operations, or 2 * B * Nparams. 
* Modern LLMs use Mixture of Experts, where the Feed Foward after the attention is split into parallel networks. There a (learnt) gate that routes each incoming token to the expert(s) that should take it. The result is then aggregated. So with MOEs a forward pass requires N active parameters instead of N parameters. Todo: insert typical numbers. Todo: is it done in parallel? 

Memory fetching and compute happen concurrently, so the time for one forward pass is the max of both (instead of the sum if it was done concurrently):

T = max ( t compute, t memory)

If the t compute > t memory, we say that the system is “compute bound”, if that’s the opposite we say it’s “memory bound”

Great, we now have all the basics covered, and we understand the two fundamental equations of this podcast!

The rest of the discussion is where Dwarkesh and Reiner actually use those two equations to make educated guesses on the parameters of LLMs at big labs.

Influence of batch size on costs

Let’s double click on the notion of batch at inference:
* At training time a batch is a subset of the training data
* At inference time, we can batch user requests together: we wait for instance 20 ms, get all the user requests at this time, and batch them together. To be exact, it’s not quite the user requests that are batched, but the sentences that need new tokens.

The more we wait to cluster user requests together, the larger the batch size we get - but does larger batch leads to economies of scale? For instance in Gemini one can pay for faster inference (which presumably is linked to lower batch size); can we imagine the opposite, and have a “Gemini slow” mode where the price per token is lower? Turns out, no: enter equation and graph.

Can we estimate the ideal batch size? In the ideal case, we have the system never idle: both memory and compute run at the same speed: solve equation.

t compute = t memory
B Nactive / flops = Ntotal / mem band 
Flops / mem band = B Nactive/ Ntotal 
=> B ~ 21k

15 ms ??? Don’t understand 

Multi GPU set ups
There are several types of GPU parallelism worth noting:
* Data parallelism: classic sharding - requests are sent to different GPU / GPU clusters and executed in parallel.
* Tensor parallelism: not discussed in this podcast, but widely used in production - the weights of each matrix are split across GPUs, each GPU does a part of the computation, then combine their output in an all to all communication.
* Pipelining: the neural network is split into sequential chunks, which are executed one after the other
* Expert parallelism: experts are distributed across GPUs

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

