# Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning (FINO)
This repository provides the official implementation of **Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning (FINO)**, accepted to ICLR 2026.

## Installation
FINO is implemented in **Python 3.9** and built using **JAX**.
To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Reproducing Results
The core implementation of FINO can be found in [agents/fino.py](agents/fino.py).

To reproduce our main results, you can use the following command:
```bash
python main.py
```
We provide a complete list of commands to reproduce the specific results presented in the paper.

### FINO on default tasks of OGBench
```bash
# OGBench humanoidmaze-medium-navigate-singletask-v0 (default: task1)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-v0 --agent.alpha=100 --agent.discount=0.995
# OGBench humanoidmaze-large-navigate-singletask-v0 (default: task1)
python main.py --env_name=humanoidmaze-large-navigate-singletask-v0 --agent.alpha=30 --agent.discount=0.995
# OGBench antmaze-large-navigate-singletask-v0 (default: task1)
python main.py --env_name=antmaze-large-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min
# OGBench antmaze-giant-navigate-singletask-v0 (default: task1)
python main.py --env_name=antmaze-giant-navigate-singletask-v0 --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995
# OGBench antsoccer-arena-navigate-singletask-v0 (default: task4)
python main.py --env_name=antsoccer-arena-navigate-singletask-v0 --agent.alpha=30 --agent.discount=0.995
# OGBench cube-double-play-singletask-v0 (default: task2)
python main.py --env_name=cube-double-play-singletask-v0 --agent.alpha=300
# OGBench puzzle-4x4-play-singletask-v0 (default: task4)
python main.py --env_name=puzzle-4x4-play-singletask-v0 --agent.alpha=1000
```

<details>
<summary><b>Click to see the full list of commands for all tasks</b></summary>

#### FINO on all tasks of OGBench
```bash
# OGBench humanoidmaze-medium-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task1-v0 --agent.alpha=100 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task2-v0 --agent.alpha=100 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task3-v0 --agent.alpha=100 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task4-v0 --agent.alpha=100 --agent.discount=0.995
python main.py --env_name=humanoidmaze-medium-navigate-singletask-task5-v0 --agent.alpha=100 --agent.discount=0.995
# OGBench humanoidmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=humanoidmaze-large-navigate-singletask-task1-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-singletask-task2-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-singletask-task3-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-singletask-task4-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=humanoidmaze-large-navigate-singletask-task5-v0 --agent.alpha=30 --agent.discount=0.995
# OGBench antmaze-large-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-large-navigate-singletask-task1-v0 --agent.alpha=10 --agent.q_agg=min
python main.py --env_name=antmaze-large-navigate-singletask-task2-v0 --agent.alpha=10 --agent.q_agg=min
python main.py --env_name=antmaze-large-navigate-singletask-task3-v0 --agent.alpha=10 --agent.q_agg=min
python main.py --env_name=antmaze-large-navigate-singletask-task4-v0 --agent.alpha=10 --agent.q_agg=min
python main.py --env_name=antmaze-large-navigate-singletask-task5-v0 --agent.alpha=10 --agent.q_agg=min
# OGBench antmaze-giant-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task1)
python main.py --env_name=antmaze-giant-navigate-singletask-task1-v0 --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task2-v0 --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task3-v0 --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task4-v0 --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995
python main.py --env_name=antmaze-giant-navigate-singletask-task5-v0 --agent.alpha=10 --agent.q_agg=min --agent.discount=0.995
# OGBench antsoccer-arena-navigate-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=antsoccer-arena-navigate-singletask-task1-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-singletask-task2-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-singletask-task3-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-singletask-task4-v0 --agent.alpha=30 --agent.discount=0.995
python main.py --env_name=antsoccer-arena-navigate-singletask-task5-v0 --agent.alpha=30 --agent.discount=0.995
# OGBench cube-double-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task2)
python main.py --env_name=cube-double-play-singletask-task1-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task2-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task3-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task4-v0 --agent.alpha=300
python main.py --env_name=cube-double-play-singletask-task5-v0 --agent.alpha=300
# OGBench puzzle-4x4-play-singletask-{task1, task2, task3, task4, task5}-v0 (default: task4)
python main.py --env_name=puzzle-4x4-play-singletask-task1-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task2-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task3-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task4-v0 --agent.alpha=1000
python main.py --env_name=puzzle-4x4-play-singletask-task5-v0 --agent.alpha=1000
```

### FINO on D4RL
```bash
# D4RL antmaze-umaze-v2
python main.py --env_name=antmaze-umaze-v2 --agent.alpha=10
# D4RL antmaze-umaze-diverse-v2
python main.py --env_name=antmaze-umaze-diverse-v2 --agent.alpha=10
# D4RL antmaze-medium-play-v2
python main.py --env_name=antmaze-medium-play-v2 --agent.alpha=10
# D4RL antmaze-medium-diverse-v2
python main.py --env_name=antmaze-medium-diverse-v2 --agent.alpha=10
# D4RL antmaze-large-play-v2
python main.py --env_name=antmaze-large-play-v2 --agent.alpha=3
# D4RL antmaze-large-diverse-v2
python main.py --env_name=antmaze-large-diverse-v2 --agent.alpha=3
# D4RL pen-cloned-v1
python main.py --env_name=pen-cloned-v1 --agent.alpha=1000 --agent.q_agg=min
# D4RL door-cloned-v1
python main.py --env_name=door-cloned-v1 --agent.alpha=1000 --agent.q_agg=min
# D4RL hammer-cloned-v1
python main.py --env_name=hammer-cloned-v1 --agent.alpha=1000 --agent.q_agg=min
# D4RL relocate-cloned-v1
python main.py --env_name=relocate-cloned-v1 --agent.alpha=10000
```
</details>

## Acknowledgments
This codebase builds upon [FQL](https://github.com/seohongpark/fql) and reference the implementation of [DACER](https://github.com/happy-yan/DACER-Diffusion-with-Online-RL).

## Citation
```
@inproceedings{
    shin2026flow,
    title={Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning},
    author={Yongjae Shin and Jongseong Chae and Jongeui Park and Youngchul Sung},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026},
    url={https://openreview.net/forum?id=6wd38R8L0Z}
}
```

## Contact
If you have any questions or inquiries, please feel free to contact:
**yongjae.shin@kaist.ac.kr**