## [IMEX-Reg: Implicit-Explicit Regularization in the Function Space for Continual Learning](https://openreview.net/forum?id=p1a6ruIZCT)
Accepted at Transactions for Machine Learning Reearch, 2024

<img width="600" alt="method_readme" src="https://github.com/NeurAI-Lab/IMEX-Reg/assets/56539375/34a8d923-43c4-49d6-98eb-7fdd81cd7644">

## Abstract
Continual learning (CL) remains one of the long-standing challenges for deep neural networks due to catastrophic forgetting of previously acquired knowledge. Although rehearsal-based approaches have been fairly successful in mitigating catastrophic forgetting, they suffer from overfitting on buffered samples and prior information loss, hindering generalization under low-buffer regimes. Inspired by how humans learn using strong inductive biases, we propose IMEX-Reg to improve the generalization performance of experience rehearsal in CL under low buffer regimes. Specifically, we employ a two-pronged implicit-explicit regularization approach using contrastive representation learning (CRL) and consistency regularization. To further leverage the global relationship between representations learned using CRL, we propose a regularization strategy to guide the classifier toward the activation correlations in the unit hypersphere of the CRL. Our results show that IMEX-Reg significantly improves generalization performance and outperforms rehearsal-based approaches in several CL scenarios. It is also robust to natural and adversarial corruptions with less task-recency bias. Additionally, we provide theoretical insights to support our design decisions further.

 ## How to run?
 
    python main.py  --seed 10  --dataset seq-tinyimg  --model imex_reg  --buffer_size 200   --load_best_args  --tensorboard --notes 'imex_reg baseline'
 
        
## Setup
Extended on Mammoth CL repo:  [Dark Experience for General Continual Learning: a Strong, Simple Baseline](https://papers.nips.cc/paper/2020/hash/b704ea2c39778f07c617f6b7ce480e9e-Abstract.html)

+ Use `./utils/main.py` to run experiments.
+ Use argument `--load_best_args` to use the best hyperparameters from the paper.

## Models
+ Implicit-Explicit Regularization (IMEX-Reg)

## Datasets

**Class-Il / Task-IL settings**

+ Sequential CIFAR-10
+ Sequential CIFAR-100
+ Sequential Tiny ImageNet


## Cite Our Work

If you find the code useful in your research, please consider citing our paper:

    @article{
       bhat2024imexreg,
       title={{IMEX}-Reg: Implicit-Explicit Regularization in the Function Space for Continual Learning},
       author={Prashant Shivaram Bhat and Bharath Chennamkulam Renjith and Elahe Arani and Bahram Zonooz},
       journal={Transactions on Machine Learning Research},
       issn={2835-8856},
       year={2024},
       url={https://openreview.net/forum?id=p1a6ruIZCT},
       note={}
    }
