## IMEX-Reg: Implicit-Explicit Regularization in the Function Space for Continual Learning
Extended on Mammoth CL repo:  [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

## How to run?
+ python main.py  --seed 10  --dataset seq-tinyimg  --model imex_reg  --buffer_size 200   --load_best_args  --tensorboard --notes 'imex_reg baseline'
 
        
## Setup

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.

## Models
+ Implicit-Explicit Regularization (IMEX-Reg)

## Datasets

**Class-Il / Task-IL settings**

+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny ImageNet

