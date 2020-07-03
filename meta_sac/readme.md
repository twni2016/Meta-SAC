# Meta-SAC
In 7th ICML AutoML workshop, 2020.

PyTorch implementation of Meta-SAC. Based on the popular [public github repo](https://github.com/pranz24/pytorch-soft-actor-critic/tree/d8ba7370e574340e9e0e9dd0276dbd2241ff3fd1) for SAC.

## Usage
The paper results can be reproduced by running:
```bash
python mainmeta.py configs/{env}.yml
```
Debug mode: set `exp_id` as `debug` to print the logging.

## Ablation studies
- Ablation study D.1: using arbitrary states, please set `meta_obj_s0` as False.
- Ablation study D.2: not using resampling, please set `resample` as False.
- Ablation study D.3: using classic Q, please set `meta_Q` as True.
