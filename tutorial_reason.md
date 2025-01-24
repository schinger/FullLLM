# 从头理解思考模型（LLM based Reasoning Model），O1，DeepSeek R1，Kimi K1.5

## 1. 简介
自OpenAI发布O1以来LLM进入了可推理（思考，Reasoning）的发展阶段，Scaling Law也由以前的training time扩展到了test time。近期，DeepSeek和Kimi团队同时发布了其思考模型R1和K1.5，两者的表现接近O1，并且都公布了完整的技术方案。本文以R1和K1.5的技术方案为基础，从头理解思考模型的原理和实现。
## 2. 什么是推理（Reasoning）
让我们先思考一下什么是推理（Reasoning）。推理是从问题到答案的过程，这个过程中，我们需要理解问题，分析问题，结合已知的知识，一步一步推导出答案。做个简单的类比的话，推理其实很像走迷宫。问题是起点，答案是终点，推理过程就是走迷宫的过程。这样来看，推理过程的中间步骤是必不可少的，思考模型正是通过COT（Chain of Thought）实现了这个过程。
## 3. 从预测下一个词到探索
预测下一个词是一个十分强大的目标。为了准确的预测出下一个词，模型需要展现出对文本的深刻理解才行。这甚至就是LLM的第一性原理，支持着LLM之前的快速发展。但是，为了进一步提升LLM的（推理）能力，仅仅通过准确的预测下一个词是不够的。我们需要让LLM去探索，去发现新的思考路径，总结经验教训。继续上面的类比，我们需要让LLM在推理的迷宫中不断的探索，增强成功的探索过程（正向梯度更新），减弱失败的探索过程（负向梯度更新）。强化学习是达成这一目标的最佳选择。
## 4. GRPO算法
PPO是TRPO的简化，GRPO是PPO的简化。这里，我们介绍下DeepSeek-R1中应用的强化学习算法GRPO（Group Relative Policy Optimization），我们直接看优化目标：
$$
\mathcal{J}_{GRPO}(\theta) = \mathbb{E}[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q) ]\\
 \qquad  \qquad  \qquad
\frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) \right),
$$


其中，
$$
A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \cdots, r_G\})}{\text{std}(\{r_1, r_2, \cdots, r_G\})}.
$$
$$
\mathbb{D}_{KL}(\pi_\theta || \pi_{ref}) = \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - \log \frac{\pi_{ref}(o_i|q)}{\pi_\theta(o_i|q)} - 1,
$$
具体来讲，GRPO对特定的问题q采样生成一组输出$\{o_1, o_2, \cdots, o_G\}$，然后得到相应的reward $\{r_1, r_2, \cdots, r_G\}$，将$r_i$标准化后得到$A_i$，取代了PPO中复杂的GAE。$\text{min}(\cdot,\text{clip}(\cdot))$的解释详见[PPO原理解读](https://zhuanlan.zhihu.com/p/3333839684)，是为了让策略更新的幅度不至于过大。

最后，GRPO中的reward不包含KL term，而是直接将其作为一个约束条件，并采用了$\mathbb{D}_{KL}(q(x)||p(x)) = \mathbb{E}_{x \sim q(x)}[\frac{p(x)}{q(x)}-\log \frac{p(x)}{q(x)}-1]$的形式。这一形式相对于原始的KL散度$\mathbb{D}_{KL}(q(x)||p(x)) = \mathbb{E}_{x \sim q(x)}[-\log \frac{p(x)}{q(x)}]$方差更小且非负，更容易优化。具体而言，首先$\mathbb{E}_{x \sim q(x)}[\frac{p(x)}{q(x)}-1]=0$，所以这一估计是unbias的，其次$ x-1 \geq \log x$，$x-1$与$-\log x$单调性相反（一定程度负相关），所以这一KL表达式方差更小且非负。

K1.5中用到的RL算法虽然是从其他角度出发但是最终的梯度形式及约束是类似的，并且同样抛弃了Value Model（Critic），在此不再赘述。
## 5. 如何实现
### 5.1 简单直接：DeepSeek-R1-Zero
DeepSeek-R1-Zero 采用了非常简单直接的做法，直接在base model（预测下一个词）的基础上进行强化学习-GRPO，效果非常好。其训练数据的形式为<question, answer>对，模型需要生成思考过程和答案，即COT+answer。Reward为答案是否正确（对于code问题为test cases是否通过）及输出是否符合格式。为了诱发模型输出COT，DeepSeek-R1-Zero采用了如下模板：
```
A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: question. Assistant:
```
这个模板十分简洁，作者也故意没有加入过多的要求，比如要求模型反思，尝试其他思路，采用特定思考或搜索策略等等，就是为了不对模型的思考过程（COT）进行过多的干预，让模型自己去探索。从名字来看DeepSeek-R1-Zero是在向AlphaZero致敬，但是一个区别是AlphaZero是完全不依赖人类经验，DeepSeek-R1-Zero最起码还需要base model。
### 5.2 DeepSeek-R1
DeepSeek-R1-Zero由于不对COT进行限制，导致COT会出现语言混杂及可读性差的问题，为了解决这一问题及进一步提升思考能力，DeepSeek-R1构建了一个包含COT的冷启动数据集<question， COT + answer>，在这个数据集上进行Supervised Finetuning（SFT） 以引导模型生成更加规范和强大的COT。总体上，DeepSeek-R1穿插着进行训练：Cold start SFT->RL->SFT->RL 最后得到了一个强大且全面的思考模型。
### 5.3 是否需要PRM，MCTS以及Value Model
R1和K1.5不一而同的未使用PRM(Process Reward Model)，MCTS以及Value Model。对于思考过程来讲，很难得到一个精准的PRM以及Value Model，例如，思考过程中间可能出现错误的方向，但是后来经过反思又修正了，对于当时的错误方向，PRM以及Value Model可能会给出负向的判断，然而这样会让模型不敢大胆尝试新的思考路径。再者，模型可以生成很多正确的中间步骤骗得Process Reward但是最终答案却错误。对于棋类游戏就不一样了，评判棋局的优劣势是一个客观问题，Value Model可以给出准确的评判。

MCTS是一个强大的决策工具，然而将MCTS应用于语言推理有个困难，语言不像棋类那样有明确的action界定，每一步的选择都太多了，MCTS的搜索空间近乎无穷。再者，任何MCTS过程都可以被拍平成一个串用COT来代替。甚至人类的思考过程按时间展开也是一个COT。COT可以是任何符号串，不一定必须是人类语言，可以是机器发明的新语言或者符号体系，所以推理能力可以超越人类。COT is all you need。

但是，思考模型不需要PRM，MCTS以及Value Model一定成立吗？我看未必，前面的讨论只是一个角度，说不定以后通过一些方法也就work了。
## 6. 展望
预测下一个词让LLM掌握了语言，RL + COT让LLM掌握了代码，数学和思考，而改进人工智能最需要的三项能力就是语言，代码和数学，LLM离改进自身已经不远了。现在需要人类来提升LLM在各个benchmark上的表现，以后多个LLM agent通过合作可能就可以自己改进和训练自己了。