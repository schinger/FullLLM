# 从头实现LLM：Pre-train, Post-train (PPO RLHF), Inference (KV Cache)

## 1. 背景介绍

2022年12月，OpenAI发布了震惊世界的ChatGPT，激起了人们对AGI的热情和讨论。其后，LLM这一技术快速发展，不仅能处理文本，还能处理图像、视频、声音等多模态数据，近期在代码和数学等需要复杂推理的领域也崭露头角。时至今日，LLM的规模越来越大，其复杂程度也越来越高，往往需要复杂的实现及优化才能高效训练，这对于想要深入学习研究LLM的朋友来说是一个挑战。然而，LLM的核心思想是简单的，只要理解了其核心思想，就能够从头实现一个LLM，[本文即是一个从头实现完整LLM的教程](https://github.com/schinger/FullLLM): https://github.com/schinger/FullLLM。

本文的实现基于karpathy的[llama2.c](https://github.com/karpathy/llama2.c)项目，它的特色是基于C语言的LLama推理。我们的项目移除了C语言相关及其它庞杂部分，保留了其中LLama的python实现，并增加了PPO和KV Cache等功能，仅依赖于Pytorch，整个项目简洁易懂，包含了LLM的完整实现：Pre-train, Post-train (PPO RLHF), Inference (KV Cache)。

## 2. Pre-train
### 2.1. 模型结构
`model.py`: 本文采用LLama3的模型结构：在传统的transformer基础上，采用pre-normalization using RMSNorm，使用SwiGLU激活，采用了RoPE相对位置编码及Group Query Attention（GQA）。当然，目前的LLM并不局限于transformer，更高效的模型结构也在不断涌现，如Mamba、RWKV等等，后续，我们会在另一篇文章中详细介绍模型结构。**由于文本，图像、视频、声音、代码、数学等等各种输入均可被token化，LLM处理它们的方式是一致的**，我们在此以文本为例来说明LLM的实现。
### 2.2. 训练目标
预训练的目标非常简单，即给定一个文本序列，预测序列的下一个词（token），具体实现采用的loss function为cross entropy。为什么预测下一个词能学到一个强大的模型呢？直观来讲，为了准确的预测出下一个词，模型需要展现出对文本的深刻理解才行。另一个角度来看，能更好的预测下一个词的模型其实是一个[无损的数据压缩器](https://www.youtube.com/watch?v=dO4TPJkeaaU)，预测的越准确，压缩率越高，压缩率越高则智能程度越高（聪明的人/机器总能用简洁的语言表达复杂的事物）。
### 2.3. 数据集
本文采用[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)数据集进行预训练，这是一个小型的英文文本数据集，包含了一些短小的儿童故事，适合作为宝宝的睡前读物。我们用`tinystories.py`下载、预处理并构建DataLoader：

```
python tinystories.py download
```
接下来我们训练一个词典大小为512的sentencepiece tokenzier，并对数据集分词处理：
```
python tinystories.py train_vocab --vocab_size=512
python tinystories.py pretokenize --vocab_size=512
```
现在我们可以开始从头预训练了，我们训练一个仅260k参数的LLM：
```
python train.py \
    --out_dir="outmini" \
    --batch_size=128 \
    --max_seq_len=512 \
    --gradient_accumulation_steps=1 \
    --vocab_source="custom" \
    --vocab_size=512 \
    --dim=64 \
    --n_layers=5 \
    --n_heads=8 \
    --n_kv_heads=4 \
    --multiple_of=4 \
    --learning_rate=1e-3 \
    --dropout=0.05 \
    --weight_decay=0.01 \
    --max_iters=100000 \
    --beta2=0.99 \
    --warmup_iters=1000 \
    --eval_interval=2000 \
    --eval_iters=100 \
    --compile=True
```
只需在A100上训练大概10分钟，就能得到一个不错的模型，虽然很小，但是它已经能够生成一些有趣的故事了。不想训练的朋友可以直接下载[训练好的模型](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)。
```
mkdir stories260K && cd stories260K
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model
cd ..

python sample.py --checkpoint=stories260K/stories260K.pt --tokenizer=stories260K/tok512.model --temperature=0.0 --max_new_tokens=500
```
生成的故事如下：

***
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
Lily and her mom went to the park to play. They played together and had fun. After a while, Lily's mom came into the room and saw the park. She was so happy and said, "Thank you, Lily! You are a good friend."
***

## 3. Post-train
后训练包括指令微调和对齐（alignment）。指令微调的损失类似于预训练（预测下一个词），只是训练数据不同，在此我们略过。对齐的目标是使模型的输出符合人类需求和价值观，方法有DPO，RLHF等。由于DPO的实现较简单，我们在此介绍更一般化的PPO RLHF的原理及实现。RLHF包含三个步骤：指令微调，奖励模型训练，PPO训练。其中，奖励模型的训练方法为pairwise supervised learning：
```python
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```
即，让chosen的reward大于rejected的reward，也较为简单，我们在此略过。只剩下PPO强化学习是我们需要详细介绍的。想要深入理解PPO原理的朋友可以参考[这篇文章](https://zhuanlan.zhihu.com/p/3333839684)。我们在此介绍PPO如何应用于LLM的训练。

### 3.1. RL in LLM
记prompt为x (长度为m，$x:=x_{1..m}$)，模型输出为y（长度为n，$y:=y_{1..n}$），那么状态s为prompt和模型当前生成的拼接，$s = (x, y_{1..t})$。动作a为模型输出的下一个token，$a = y_{t+1}$。奖励r为reward函数/模型输出，$r = r(x, y)$，策略为$\pi_\theta(y_{t+1}|x, y_{1..t})$，生成整个y的概率为$\pi_\theta(y|x) = \prod_{t=0}^{n-1} \pi_\theta(y_{t+1}|x, y_{1..t})$。RL最大化的目标为：


$$
\text{objective}(\theta) = \mathbb{E}_{(x, y) \sim D_{\pi_\theta}} \left[ r(x, y) - \beta \log(\pi_\theta(y \mid x) / \pi^{REF}(y \mid x)) \right]
$$


其中，$\pi^{REF}$为参考策略，即经过指令微调的模型。为了保证RL策略和参考策略差异不要太大，在原始reward的基础上增加了两者的KL散度惩罚项：$ \beta \log(\pi_\theta^{RL}(y \mid x) / \pi^{REF}(y \mid x))$。可以看出，只有模型生成了最后一个token，y才完整，才能得到reward，生成中间token时reward为0。然而，我们可以对KL项进行拆解，将其分解到每个token的头上：
$$
\begin{aligned}
\log(\pi_\theta^{RL}(y \mid x) / \pi^{REF}(y \mid x)) 
&=  \log(\prod_{t=0}^{n-1} \pi_\theta(y_{t+1}|x, y_{1..t}) / \pi^{REF}(y_{t+1}|x, y_{1..t})) \\
&= \sum_{t=0}^{n-1} \log(\pi_\theta(y_{t+1}|x, y_{1..t}) / \pi^{REF}(y_{t+1}|x, y_{1..t})) \\
&= \sum_{t=0}^{n-1} [\log(\pi_\theta(y_{t+1}|x, y_{1..t})) - \log(\pi^{REF}(y_{t+1}|x, y_{1..t}))]
\end{aligned} 
$$
这样，每个token都能收到reward，最后一个token收到两种reward之和，具体对应代码中`config.kl_penalty == "kl"`时，`compute_rewards`方法返回的第一项。对于KL还可以有不同的理解和实现，具体请参考代码。

最后，在进行RL训练时，还可以加入一些预训练的数据和损失，以防模型对预训练知识的遗忘。这一节的论述其实是general的RL in LLM，不仅适用于PPO，也适用于其他RL算法，如Reinforce等。


### 3.2. 实现方案

- `train.py`: 训练主循环
- `model.py`: `TransformerWithValueHead`类，同时输出策略和value （通过`self.value_head = nn.Linear(params.dim, 1)`将隐层转化成scalr value）
- `ppo.py`: `PPOConfig`类和`PPOTrainer`类，分别用于配置PPO参数和训练PPO模型。

#### 3.2.1. PPO 训练主循环

参考`train.py` 376-406行：
1. 准备个数为`batch_size`的prompt/query数据。
2. 调用`model.generate_withcache`采样生成response （在RL里一般称为rollout）。
3. 调用reward model得到reward（这里不包含KL penalty）
4. 调用`ppo_trainer.step(query_tensors, response_tensors, reward_tensors)`执行一个训练步。

#### 3.2.2. PPOTrainer.step
1. 再次对(query, response)调用model forward，得到logprobs和values
2. 对(query, response)调用reference model forward, 得到ref_logprobs
3. 按照3.1节的方法`compute_rewards`，即，包含了KL penalty的rewards
4. 计算advantages，即$\sum_{l=0}^{\infty} (\gamma \lambda)^l \delta^{V}_{t+l}$， 其中$\delta_t^{V} = r_t + \gamma V(s_{t+1}) - V(s_t)$，并计算returns = advantages + values
5. ppo training loop:
```python
for ep in ppo_epochs:
    for backward_batch in batch:
        for mini_batch in backward_batch:
            # call current model forward
            logprobs, logits, vpreds, _ = batched_forward_pass(model, mini_batch)
            old_logprobs, old_values, advantages, returns = mini_batch
            # compute loss
            loss, train_stats = compute_loss(old_logprobs, old_values, logprobs, vpreds, advantages, returns)
            # gradient_accumulation_steps and backward
            require_backward_grad_sync()
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()

def compute_loss(old_logprobs, old_values, logprobs, vpreds, advantages, returns):
    vpredclipped = clip_by_value(
            vpreds,
            old_values - cliprange_value,
            old_values + cliprange_value,
        )
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    vf_loss = 0.5 * masked_mean(torch.max(vf_losses1, vf_losses2), mask)

    ratio = torch.exp(logprobs - old_logprobs)
    pg_losses = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    pg_loss = masked_mean(torch.max(pg_losses, pg_losses2), mask)

    loss = pg_loss + vf_coef * vf_loss
    stats = ...
    return loss, stats
```
可以看出，上面的loss不仅对policy做了clip，还对value做了clip。在这里，我们省略了统计量stats的计算，这些stats对于debug和分析非常有用，可以用于监控训练过程，具体请参考代码。它们的含义如下：
```
{
    "ppo/loss/policy": policy loss,
    "ppo/loss/value": value loss,
    "ppo/loss/total": total loss,
    "ppo/policy/entropy": policy entropy,
    "ppo/policy/policykl": old and current policy kl,
    "ppo/policy/clipfrac": clip fraction of policy,
    "ppo/policy/advantages_mean": advantages mean,
    "ppo/returns/mean": returns mean,
    "ppo/returns/var": returns variance,
    "ppo/val/vpred": predicted value,
    "ppo/val/error": mean of (vpreds - returns) ** 2,
    "ppo/val/clipfrac": clip fraction of value,
    "ppo/val/mean": value mean,
    "ppo/val/var": value variance,
    "ppo/learning_rate": lr,
    "ppo/kl_ref": refrerence model and current model kl,
    "ppo/reward_all": the final reward
}
```

### 3.3. 本文实例
本文继续以TinyStories为例进行PPO训练，prompt/query为TinyStories中的起始几个词，训练目标为完成故事至指定的长度，我们令reward为: r = -abs(target_length - story_length)/100。

我们将Pre-train小节训练好或者[下载](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K)好的model重命名为ckpt.pt，执行命令：
```
python train.py     \
    --out_dir="stories260K"     \
    --batch_size=50     \
    --max_seq_len=512     \
    --gradient_accumulation_steps=1     \
    --vocab_source="custom"     \
    --vocab_size=512     \
    --dim=64     \
    --n_layers=5     \
    --n_heads=8     \
    --n_kv_heads=4     \
    --multiple_of=4     \
    --learning_rate=1e-4     \
    --dropout=0.00     \
    --weight_decay=0.01     \
    --max_iters=98049     \
    --beta2=0.99     \
    --warmup_iters=1000     \
    --eval_interval=20     \
    --eval_iters=5     \
    --compile=True    \
    --device=cuda    \
    --eval_only=False   \
    --init_from="resume" \
    --ppo=True  \
    --decay_lr=False  \
    --always_save_checkpoint=True  \
    --start_len=30
```
训练大概十分钟左右，我们就可以将平均长度由原来的~300变为目标target_length 200左右。用训练好的模型生成新故事：
```
python sample.py --checkpoint=stories260K/ppo_ckpt.pt --tokenizer=stories260K/tok512.model --temperature=0.4 --max_new_tokens=500
```
生成的故事如下：
Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily's mom asked her to clean up her toys and they went to the park to play. Lily was very happy and wanted to play with her toys.
As they were walking, Lily saw a big, red balloon in the park. She asked her mom, "What's wrong, mom?" Her mom said, "I'm sorry, Lily. I didn't know what to do."
Lily said, "I don't know, I will help you." Her mom smiled and said, "Yes, Lily. You are a good friend."


总长度218，还不错，毕竟只是一个大小为260k的模型。
## 4. Inference
### 4.1. KV Cache
Inference的入口为`sample.py`，它调用了`model.py`中的`generate_withcache`方法，该方法使用了KV Cache来加速推理。KV Cache用于存储模型的中间结果（transformer中的K，V矩阵），以便在推理时复用。我们将KV Cache放在`Attention`类中，用zero tensor来初始化。
```python
self.cache_k = torch.zeros(
(
    args.max_batch_size,
    args.max_seq_len,
    self.n_local_kv_heads,
    self.head_dim,
))
self.cache_v = torch.zeros(
(
    args.max_batch_size,
    args.max_seq_len,
    self.n_local_kv_heads,
    self.head_dim,
))
```
设当前处理序列的序列长度为`seqlen`，起始位置为`start_pos`，计算出的K，V值为xk，xv，我们将其存入KV Cache中：
```python
self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv
```
在推理时，我们可以直接从KV Cache中取出所有K，V值，而不需要再次计算：
```python
xk = self.cache_k[:bsz, : start_pos + seqlen]
xv = self.cache_v[:bsz, : start_pos + seqlen]
```
### 4.2. TopK, TopP采样
另外，`model.py`中还实现了topK采样：只保留概率最大的K个token，其余的token概率置为0，然后再归一化采样。topP采样：只保留概率之和大于P的最少token，其余的token概率置为0，然后再归一化采样。
```python
def sample_top_k_top_p(logits, top_k=0, top_p=0.0, temperature=1.0):
    logits = logits / temperature
    if top_k > 0:
        # optionally crop the logits to only the top k options
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    if 0 <= top_p <= 1.0:
        # Top-p (nucleus) sampling selects the smallest set of tokens whose cumulative probability mass
        # exceeds the threshold p. The distribution is renormalized based on the selected tokens.
        probs = F.softmax(logits, dim=-1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
```
## 5. o1如何实现
上文论述了给定Prompt/Query，如何用RL训练模型生成Response/Answer。o1的setting是给定Prompt/Query，生成COT + Response/Answer。所以，用RL训练时，两者没有本质的区别，方法可以直接照搬。


o1需要额外生成中间COT，另外，reward的设计也有所不同，o1的reward可以是只检查Answer的正确性(Outcome-supervised Reward)，或者也检查中间COT过程的正确性（Process-supervised Reward）。具体可参考[这篇文章](https://zhuanlan.zhihu.com/p/721333377)。
## 6. 展望

一个有趣的方向是如何让LLM自我提升，甚至自己训练自己，最终超越人类的智能。AlphaZero是个很好的例子，可以参考这篇文章：[从头理解AlphaZero，MCTS，Self-Play，UCB](https://zhuanlan.zhihu.com/p/5367995214)