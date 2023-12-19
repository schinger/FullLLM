## Fullstack LLM (Llama2): Pre-training/finetuning, PPO(RLHF), Inference, Int8 Quantization

[Karpathy/llama2.c](https://github.com/karpathy/llama2.c) is a great project that includes nearly all the parts to build LLMs. The only thing it doesn't include is RLHF. This project is based on llama2.c and add PPO(RLHF) to it. Also, we make README.md more clear and follow the steps to build LLMs from scratch.

## Custom tokenizers
Firstly, we use [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset throughout the whole project. To download:
```
python tinystories.py download
```
To train an example 512-token tokenizer:
```
python tinystories.py train_vocab --vocab_size=512
python tinystories.py pretokenize --vocab_size=512
```
The `train_vocab` stage will call the `sentencepiece` library to train the tokenizer, storing it in a new file `data/tok512.model`. The `pretokenize` stage here loads the trained tokenizer and uses it to convert the downloaded text into integers, and saves that to file.

You can also leverage the default Llama2 tokenizers (vocab size 32,000, we don't use it in this project):
```
python tinystories.py pretokenize
```
## Pre-training/finetuning
We pretrain a very small Llama2 model (260K parameters) from [scratch](https://github.com/karpathy/llama2.c/blob/master/doc/stories260K.md):
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
It only need ~10 minutes to finish the training on a single A100 and achieves validation loss of 1.2968. The ckpt is [here](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K). Though it's a very small model, it can generate somehow fluent sentences:
```
mkdir stories260K && cd stories260K
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.pt
wget https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.model
cd ..

python sample.py --checkpoint=stories260K/stories260K.pt --tokenizer=stories260K/tok512.model --temperature=0.0 --max_new_tokens=500
```
```
Once upon a time, there was a little girl named Lily. She loved to play outside in the park. One day, she saw a big, red ball. She wanted to play with it, but it was too high.
Lily's mom said, "Lily, let's go to the park." Lily was sad and didn't know what to do. She said, "I want to play with your ball, but I can't find it."
Lily was sad and didn't know what to do. She said, "I'm sorry, Lily. I didn't know what to do."
Lily didn't want to help her mom, so she said, "I'm sorry, mom. I didn't know what to do." Her mom said, "Don't worry, Lily. We can help you."
Lily and her mom went to the park to play. They played together and had fun. After a while, Lily's mom came into the room and saw the park. She was so happy and said, "Thank you, Lily! You are a good friend."
```

## RLHF training
The three steps of RLHF training are: supervised fine-tuning, reward model training, PPO. Supervised fine-tuning are the same as pre-training except for data format, so we skip it here. The reward model training is supervised pairwise learning:

```python
loss = -nn.functional.logsigmoid(rewards_chosen - rewards_rejected).mean()
```

So we also skip it. The PPO training is the core of RLHF training. Here we use a very simple reward function (token length of story close to a target length) to show how to use PPO to train a LLM. The reward function is defined in `ppo.py`:

```python
class LengthReward:
    """
    Calculates the reward based on the length of the sequence
    """

    def __init__(self, target_length=200):
        self.target_length = target_length
    
    def __call__(self, sequence_length):
        return -abs(self.target_length - sequence_length)/100.
```

The prompt is some start words of the story in the training data. The goal of PPO is to complete the story close to the target length. 
To directly start PPO training, download the pretrained model [here](https://huggingface.co/karpathy/tinyllamas/tree/main/stories260K), rename `stories260K.pt` to `ckpt.pt` and run:


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

Initial statistics:
```
q+r lengths [252, 297, 328, 328, 328, 328, 328, 328, 328, 328]
q+r lengths avg 304.1
98000 | reward -1.0687 | lr 1.000000e-04 | 280442.50ms
{
    "ppo/loss/policy": -0.014800487086176872,
    "ppo/loss/value": 0.060744259506464005,
    "ppo/loss/total": -0.008726062253117561,
    "ppo/policy/entropy": 1.3361481428146362,
    "ppo/policy/policykl": 0.004431259818375111,
    "ppo/policy/clipfrac": 0.04800604283809662,
    "ppo/policy/advantages_mean": 0.00011433474719524384,
    "ppo/returns/mean": -0.08783192932605743,
    "ppo/returns/var": 0.04501805081963539,
    "ppo/val/vpred": -0.04483073577284813,
    "ppo/val/error": 0.12132434546947479,
    "ppo/val/clipfrac": 0.00803137756884098,
    "ppo/val/mean": -0.019085045903921127,
    "ppo/val/var": 0.10107368230819702,
    "ppo/learning_rate": 0.0001,
    "ppo/kl_ref": 0.0,
    "ppo/reward_all": -1.0686500072479248,
    "time/ppo/forward_pass": 28.0527503490448,
    "time/ppo/compute_rewards": 0.046872615814208984,
    "time/ppo/compute_advantages": 0.031641483306884766,
    "time/ppo/optimize_step": 103.44326996803284,
    "time/ppo/calc_stats": 0.015612363815307617,
    "time/ppo/total": 131.59014678001404
}
```
In the begining the avg length is 304.1, which is far from the target length 200. After only 49 iterations (~10 minutes), the avg length can reach 200.235:

```
q+r lengths [169, 230, 199, 135, 197, 172, 297, 189, 162, 201]
q+r lengths avg 200.235
98049 | reward -0.3672 | lr 1.000000e-04 | 212372.84ms
{
    "ppo/loss/policy": -0.015469990670681,
    "ppo/loss/value": 0.03054601326584816,
    "ppo/loss/total": -0.012415390461683273,
    "ppo/policy/entropy": 1.2525558471679688,
    "ppo/policy/policykl": 0.004105564206838608,
    "ppo/policy/clipfrac": 0.045593008399009705,
    "ppo/policy/advantages_mean": 0.0001622058916836977,
    "ppo/returns/mean": -1.400254726409912,
    "ppo/returns/var": 0.0390283428132534,
    "ppo/val/vpred": -1.3735605478286743,
    "ppo/val/error": 0.06109202653169632,
    "ppo/val/clipfrac": 0.0,
    "ppo/val/mean": -1.363938331604004,
    "ppo/val/var": 0.04070740565657616,
    "ppo/learning_rate": 0.0001,
    "ppo/kl_ref": 7.249481201171875,
    "ppo/reward_all": -1.81709623336792,
    "time/ppo/forward_pass": 26.893563508987427,
    "time/ppo/compute_rewards": 0.04730224609375,
    "time/ppo/compute_advantages": 0.031235456466674805,
    "time/ppo/optimize_step": 97.83660507202148,
    "time/ppo/calc_stats": 0.0,
    "time/ppo/total": 124.80870628356934
}
```
set temperature to 0.4, it can generate fluent sentences (total length: 218):
```
python sample.py --checkpoint=stories260K/ppo_ckpt.pt --tokenizer=stories260K/tok512.model --temperature=0.4 --max_new_tokens=500
```
```
Once upon a time, there was a little girl named Lily. She loved to play outside in the park with her friends. One day, Lily's mom asked her to clean up her toys and they went to the park to play. Lily was very happy and wanted to play with her toys.
As they were walking, Lily saw a big, red balloon in the park. She asked her mom, "What's wrong, mom?" Her mom said, "I'm sorry, Lily. I didn't know what to do."
Lily said, "I don't know, I will help you." Her mom smiled and said, "Yes, Lily. You are a good friend."
```
What if we just want to fit the target length and don't care about the quality of the story? set `--init_kl_coef=0.0` which remove the KL penalty term in PPO loss. 

### Just learn the target length

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
    --max_iters=200000     \
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
    --start_len=1  \
    --init_kl_coef=0.0
```
after run several iterations, it can reach the length goal. But achieve a huge KL with the reference model(pretrained model) and generate meaningless sentences.
```
q+r lengths [201, 200, 200, 201, 199, 199, 200, 201, 200, 200]
q+r lengths avg 200.175
98147 | reward -0.0087 | lr 1.000000e-04 | 154092.34ms
{
    "ppo/loss/policy": -0.004611059091985226,
    "ppo/loss/value": 7.655927038285881e-05,
    "ppo/loss/total": -0.004603402689099312,
    "ppo/policy/entropy": 0.36575931310653687,
    "ppo/policy/policykl": 0.0045680576004087925,
    "ppo/policy/clipfrac": 0.022210117429494858,
    "ppo/policy/advantages_mean": 0.00017487886361777782,
    "ppo/returns/mean": -0.008375758305191994,
    "ppo/returns/var": 0.00014193591778166592,
    "ppo/val/vpred": -0.008607665076851845,
    "ppo/val/error": 0.00015311854076571763,
    "ppo/val/clipfrac": 0.0,
    "ppo/val/mean": -0.008562136441469193,
    "ppo/val/var": 3.553760689101182e-05,
    "ppo/learning_rate": 0.0001,
    "ppo/kl_ref": 609.431640625,
    "ppo/reward_all": -0.008749999105930328,
    "time/ppo/forward_pass": 22.32808017730713,
    "time/ppo/compute_rewards": 0.04999852180480957,
    "time/ppo/compute_advantages": 0.029018878936767578,
    "time/ppo/optimize_step": 81.42295384407043,
    "time/ppo/calc_stats": 0.004998207092285156,
    "time/ppo/total": 103.83504962921143
}
```
## C Inference and Int8 Quantization
See [Karpathy/llama2.c](https://github.com/karpathy/llama2.c) 

## TODO
- todo
## License

MIT
