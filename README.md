# RLTutor
This repository is the official code introduced in our paper: [RLTutor: Reinforcement Learning Based Adaptive Tutoring System by Modeling Virtual Student with Fewer Interactions](https://arxiv.org/abs/2108.00268), AI4EDU workshop at IJCAI 2021.

## Prerequisites
After cloning this repository, it is required to work though the following two instructions to run the experiment.

### 1. Library Installation
For running our code, the following libraries are required.
 - python = "^3.8"
 - gym = "^0.18.0"
 - matplotlib = "^3.3.4"
 - numpy = "^1.20.1"
 - pandas = "^1.2.1"
 - pfrl = "^0.2.1"
 - scikit-learn = "^0.24.1"
 - scipy = "^1.6.0"
 - torch = "1.8.1"
 - hydra-core = "^1.1.0"

Note that torch version is very sensitive to which version of cuda you are using.  Please change it according to your local environment.

If you use [poetry](https://python-poetry.org/) as the package manager, all you have to do after cloning is:

```poetry install --no-dev```


### 2. Pretrained Data Preparation

We used [DAS3H](https://github.com/BenoitChoffin/das3h)-based model as the inner model, and it was pretrained using [EdNet](https://github.com/riiid/ednet) data.
Pretrained weight is included in the data folder, so you can reproduce the results on the paper by using it.
However, if you try to pretrain with the other dataset, you have to train the DAS3H before running our experiment.

## Conduct Experiment
To reproduce the results of the paper, you can first run the following code at the project root:

```python src/framework.py -m seed=1,2,3,4,5```

After finishing the experiment, you can plot the result following the instruction in Colab Notebook.