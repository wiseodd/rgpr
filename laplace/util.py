import torch
import torch.distributions as dist
from tqdm import tqdm
import rgpr.kernel as rgp_kernel
from time import time
from tqdm import tqdm, trange
import numpy as np
import swag.utils


@torch.no_grad()
def predict(test_loader, model, n_samples=10, n_classes=10, apply_softmax=True, return_targets=False, delta=1, rgpr=False, base_model=None, kernel_vars=None, means=None, stds=None):
    py = []
    targets = []

    for x, y in test_loader:
        x, y = delta*x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        targets.append(y)

        if rgpr:
            mean = torch.zeros(x.shape[0], n_classes, device='cuda')

            x_ = x.flatten(1)

            # Centering
            if means is not None:
                assert stds is not None
                x_ = (x_ - means[0][None, :])/stds[0][None, :]

            var = rgp_kernel.kernel(x_, x_, kernel_vars[0]).reshape(-1, 1)

            _, acts = base_model.forward(x, return_acts=True)

            for a, kernel_var, a_mean, a_std in zip(acts, kernel_vars[1:], means[1:], stds[1:]):
                a_ = a.flatten(1).detach()

                if means is not None:
                    a_ = (a_ - a_mean[None, :])/a_std[None, :]

                var += rgp_kernel.kernel(a_, a_, kernel_var).reshape(-1, 1)

            var[torch.isinf(var)] = 1e31
            var[torch.isnan(var)] = 1e31
            var += 1e-8

            # Covariance matrices for the whole batch
            # C is (batch_size, num_classes, num_classes)
            C = []
            for v in var:
                C_ = v*torch.eye(n_classes, device='cuda')
                C.append(C_)
            C = torch.stack(C)

            # Hack, to mitigate a bug
            # https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624/7
            scale_tril = torch.cholesky(C.cpu()).cuda()

            p_f_hat = dist.MultivariateNormal(mean, scale_tril=scale_tril)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)

            if rgpr:
                f_hat = p_f_hat.sample()
                out += f_hat

            py_ += torch.softmax(out, 1) if apply_softmax else out

        py_ /= n_samples
        py.append(py_)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict2(test_loader, model, n_classes=10, n_samples=10, delta=1, scale=1, train_loader=None, rgpr=False, base_model=None, kernel_vars=None, means=None, stds=None):
    dset = test_loader.dataset
    predictions = torch.zeros((len(dset), n_classes))
    targets = np.zeros(len(dset))

    for _ in range(n_samples):
        k = 0

        model.sample(scale)

        if train_loader is not None:  # For SWAG
            swag.utils.bn_update(train_loader, model)

        model.eval()

        for x, y in test_loader:
            x = delta*x.cuda(non_blocking=True)

            out = model(x)

            if rgpr:
                mean = torch.zeros(x.shape[0], n_classes, device='cuda')

                x_ = x.flatten(1)

                # Centering
                if means is not None:
                    assert stds is not None
                    x_ = (x_ - means[0][None, :])/stds[0][None, :]

                var = rgp_kernel.kernel(x_, x_, kernel_vars[0]).reshape(-1, 1)

                _, acts = base_model.forward(x, return_acts=True)

                for a, kernel_var, a_mean, a_std in zip(acts, kernel_vars[1:], means[1:], stds[1:]):
                    a_ = a.flatten(1).detach()

                    if means is not None:
                        a_ = (a_ - a_mean[None, :])/a_std[None, :]

                    var += rgp_kernel.kernel(a_, a_, kernel_var).reshape(-1, 1)

                var[torch.isinf(var)] = 1e35
                var[torch.isnan(var)] = 1e35
                var += 1e-8

                # Covariance matrices for the whole batch
                # C is (batch_size, num_classes, num_classes)
                C = []
                for v in var:
                    C_ = v*torch.eye(n_classes, device='cuda')
                    C.append(C_)
                C = torch.stack(C)

                # Hack, to mitigate a bug
                # https://discuss.pytorch.org/t/cuda-illegal-memory-access-when-using-batched-torch-cholesky/51624/7
                scale_tril = torch.cholesky(C.cpu()).cuda()

                p_f_hat = dist.MultivariateNormal(mean, scale_tril=scale_tril)

                f_hat = p_f_hat.sample()
                out += f_hat

            predictions[k:k+x.shape[0]] += torch.softmax(out, dim=1).cpu().numpy()
            k += x.shape[0]

    predictions /= n_samples

    return predictions


@torch.no_grad()
def predict2_reg(X_test, model, n_samples=100, scale=1, rgpr=False, base_model=None, kernel_var=1):
    dset = X_test.cuda()
    predictions = torch.zeros(n_samples, len(X_test))

    for s in range(n_samples):
        k = 0

        model.sample(scale)
        model.eval()

        out = model(X_test)

        if rgpr:
            x_ = X_test.flatten(1)
            var = rgp_kernel.kernel(x_, x_, kernel_var).reshape(-1, 1)

            _, acts = base_model.forward(X_test, return_acts=True)

            for a in acts:
                a = a.flatten(1).detach()
                var += rgp_kernel.kernel(a, a, kernel_var).reshape(-1, 1)

            var[torch.isinf(var)] = 1e31
            var[torch.isnan(var)] = 1e31
            var += 1e-8

            out += var.sqrt()*torch.randn(*var.shape, device='cuda')

        predictions[s, :] = out.squeeze().cpu()

    return predictions.mean(0), predictions.var(0)


@torch.no_grad()
def predict_binary(test_loader, model, n_samples=100, return_targets=False, delta=1, rgpr=False):
    py = []
    targets = []

    for x, y in test_loader:
        x, y = delta*x.cuda(), y.cuda()
        targets.append(y)

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x).squeeze()
            py_ += torch.sigmoid(out)

        py_ /= n_samples
        py.append(py_)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)
