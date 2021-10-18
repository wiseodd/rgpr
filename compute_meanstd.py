import torch
import numpy as np
import util.dataloaders as dl
from tqdm import tqdm, trange
from sklearn.preprocessing import StandardScaler
from models import models, resnet_orig


num_classes = lambda dataset: 100 if dataset == 'CIFAR100' else 10
num_layers = lambda dataset: 4 if dataset == 'MNIST' else 5


def load_model(dataset):
    if dataset == 'MNIST':
        model = models.LeNetMadry().cuda()
    else:
        model = resnet_orig.ResNet18(num_classes=num_classes(dataset)).cuda()

    model.load_state_dict(torch.load(f'./pretrained_models/{dataset}_plain.pt'))
    model.eval()

    return model


means, stds = {}, {}

for dataset in ['MNIST', 'CIFAR10', 'SVHN', 'CIFAR100']:
    print(f'Processing {dataset}')

    model = load_model(dataset)
    scalers = [StandardScaler() for _ in range(num_layers(dataset))]
    train_loader = dl.datasets_dict[dataset](train=True, augm_flag=False)

    for x, _ in train_loader:
        with torch.no_grad():
            _, acts = model.forward(x.cuda(), return_acts=True)

        # For the actual data
        scalers[0] = scalers[0].partial_fit(x.flatten(1).numpy())

        # For activations
        for i, act in enumerate(acts):
            scalers[i+1] = scalers[i+1].partial_fit(act.flatten(1).cpu().numpy())

    means[dataset] = [scaler.mean_ for scaler in scalers]
    stds[dataset] = [scaler.scale_ for scaler in scalers]

np.save('means.npy', means)
np.save('stds.npy', means)
print('Done!')
