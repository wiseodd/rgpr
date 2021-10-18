#!/bin/bash

# =====================================================================
# Hyperparams search
# =====================================================================

python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --dataset MNIST
python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --dataset SVHN
python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --dataset CIFAR10
python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --dataset CIFAR100

python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --ood_dset imagenet --dataset MNIST
python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --ood_dset imagenet --dataset SVHN
python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --ood_dset imagenet --dataset CIFAR10
python eval_OOD.py --optimize_hyper --dont_save --lam 0.5 --ood_dset imagenet --dataset CIFAR100



# =====================================================================
# Far-away confidence
# =====================================================================

for i in {1..3}
do
    python eval_OOD.py --dataset MNIST --randseed $((RANDOM)) --faraway
done

for i in {1..3}
do
    python eval_OOD.py --dataset CIFAR10 --randseed $((RANDOM)) --faraway
done

for i in {1..3}
do
    python eval_OOD.py --dataset SVHN --randseed $((RANDOM)) --faraway
done

for i in {1..3}
do
    python eval_OOD.py --dataset CIFAR100 --randseed $((RANDOM)) --faraway
done


# =====================================================================
# OOD detection
# =====================================================================

for i in {1..5}
do
    python eval_OOD.py --dataset MNIST --randseed $((RANDOM))
done

for i in {1..5}
do
    python eval_OOD.py --dataset CIFAR10 --randseed $((RANDOM))
done

for i in {1..5}
do
    python eval_OOD.py --dataset SVHN --randseed $((RANDOM))
done

for i in {1..5}
do
    python eval_OOD.py --dataset CIFAR100 --randseed $((RANDOM))
done


# =====================================================================
# Rotated-MNIST
# =====================================================================

for i in {1..5}
do
  python eval_rotMNIST.py --randseed $((RANDOM))
done


# =====================================================================
# CIFAR-10-C
# =====================================================================

python eval_cifar10c.py


# =====================================================================
# OOD detection --- ImageNet
# =====================================================================

for i in {1..5}
do
    python eval_OOD.py --dataset MNIST --ood_dset imagenet --randseed $((RANDOM))
done

for i in {1..5}
do
    python eval_OOD.py --dataset CIFAR10 --ood_dset imagenet --randseed $((RANDOM))
done

for i in {1..5}
do
    python eval_OOD.py --dataset SVHN --ood_dset imagenet --randseed $((RANDOM))
done

for i in {1..5}
do
    python eval_OOD.py --dataset CIFAR100 --ood_dset imagenet --randseed $((RANDOM))
done


# =====================================================================
# Regression exps --- far-away variance
# =====================================================================

for i in {1..10}
do
    python eval_reg.py --randseed $((RANDOM))
done


# =====================================================================
# Generating figures
# =====================================================================

python generate_graph.py --dataset MNIST
python generate_graph.py --dataset MNIST --no_rgpr
python generate_graph.py --dataset CIFAR10
python generate_graph.py --dataset CIFAR10 --no_rgpr
python generate_graph.py --dataset SVHN
python generate_graph.py --dataset SVHN --no_rgpr
python generate_graph.py --dataset CIFAR100
python generate_graph.py --dataset CIFAR100 --no_rgpr
