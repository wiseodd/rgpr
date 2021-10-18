import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.utils import shuffle as skshuffle
import gpytorch
import rgpr.kernel as rgp_kernel
import torch.distributions as dist


@torch.no_grad()
def predict(dataloader, model, n_samples=1, T=1, delta=1, return_targets=False):
    py = []
    targets = []

    for x, y in dataloader:
        x = delta*x.cuda()

        py_ = 0
        for _ in range(n_samples):
            f_s = model.forward(x)
            py_ += torch.softmax(f_s/T, 1)
        py_ /= n_samples

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_ensemble(dataloader, models, T=1, delta=1, return_targets=False):
    py = []
    targets = []

    for x, y in dataloader:
        x = delta*x.cuda()

        py_ = 0
        for model in models:
            f_s = model.forward(x)
            py_ += 1/len(models) * torch.softmax(f_s/T, 1)

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_logit(dataloader, model):
    logits = []

    for x, _ in dataloader:
        x = x.cuda()
        out = model.forward(x)
        logits.append(out)

    return torch.cat(logits, dim=0)


@torch.no_grad()
def predict_binary(dataloader, model, n_samples=1, T=1, apply_sigm=True, return_targets=False, delta=1):
    py = []
    targets = []

    for x, y in dataloader:
        x = delta * x.cuda()

        f_s = model.forward(x).squeeze()
        py_ = torch.sigmoid(f_s/T) if apply_sigm else f_s/T

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0).float()
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_blight(dataloader, model, likelihood, n_samples=20, delta=1):
    num_classes = likelihood.num_classes

    with gpytorch.settings.num_likelihood_samples(n_samples):
        py = []
        n_samples = torch.Size((n_samples,))

        for x, _ in dataloader:
            x = delta*x.cuda()

            pf = model(x.flatten(1))
            f = pf.rsample(n_samples)
            py_ = torch.softmax(f, dim=-1).mean(0)
            py.append(py_)

        py = torch.cat(py, dim=0)

    return py


@torch.no_grad()
def predict_dkl(dataloader, model, likelihood, n_samples=20, delta=1, rgpr=False, kernel_vars=None, means=None, stds=None):
    num_classes = likelihood.num_classes

    with gpytorch.settings.num_likelihood_samples(n_samples):
        py = []
        n_samples = torch.Size((n_samples,))

        for x, _ in dataloader:
            x = delta*x.cuda()

            # This gives us n samples from the predictive distribution
            f = model(x).rsample(n_samples)
            f = f @ likelihood.mixing_weights.t()
            # output = likelihood(f)

            if rgpr:
                mean = torch.zeros(x.shape[0], num_classes, device='cuda')

                x_ = x.flatten(1)

                # Centering
                if means is not None:
                    assert stds is not None
                    x_ = (x_ - means[0][None, :])/stds[0][None, :]

                var = rgp_kernel.kernel(x_, x_, kernel_vars[0]).reshape(-1, 1)

                _, acts = model.feature_extractor(x, return_acts=True)

                for a, kernel_var, a_mean, a_std in zip(acts, kernel_vars[1:], means[1:], stds[1:]):
                    a_ = a.flatten(1).detach()

                    if means is not None:
                        a_ = (a_ - a_mean[None, :])/a_std[None, :]

                    var += rgp_kernel.kernel(a_, a_, kernel_var).reshape(-1, 1)

                var[torch.isinf(var)] = 1e35
                var[torch.isnan(var)] = 1e35
                var += 1e-8
                # print(var); input()

                C = torch.diag_embed(var.repeat(1, num_classes), offset=0, dim1=-2, dim2=-1).contiguous().cpu()
                scale_tril = torch.cholesky(C).cuda()
                p_f_hat = dist.MultivariateNormal(mean, scale_tril=scale_tril)

                f_hat = p_f_hat.rsample(n_samples)
                # print(f_hat); input()
                f += f_hat

            # Taking the mean over all of the sample we've drawn
            py_ = torch.softmax(f, dim=-1).mean(0)
            # py_ = likelihood(f).probs.mean(0)
            py.append(py_)

        py = torch.cat(py, dim=0)

    return py


@torch.no_grad()
def predict_dkl_reg(X_test, model, rgpr=False, kernel_var=1):
    # This gives us n samples from the predictive distribution
    p_f = model(X_test)
    mean_pred, var_pred = p_f.mean, p_f.variance

    if rgpr:
        x_ = X_test.flatten(1)
        var = rgp_kernel.kernel(x_, x_, kernel_var).reshape(-1, 1)

        var[torch.isinf(var)] = 1e31
        var[torch.isnan(var)] = 1e31
        var += 1e-8

        var_pred += var.squeeze()

    return mean_pred, var_pred


def get_confidence(py, binary=False):
    return py.max(1) if not binary else np.maximum(py, 1-py)


def get_auroc(py_in, py_out):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return roc_auc_score(labels, examples)


def get_auroc_binary(py_in, py_out):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    conf_in = np.maximum(py_in, 1-py_in)
    conf_out = np.maximum(py_out, 1-py_out)
    examples = np.concatenate([conf_in, conf_out])
    return roc_auc_score(labels, examples)


def get_aupr(py_in, py_out):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    prec, rec, thresh = precision_recall_curve(labels, examples)
    aupr = auc(rec, prec)
    return aupr


def get_fpr95(py_in, py_out):
    conf_in, conf_out = py_in.max(1), py_out.max(1)
    tpr = 95
    perc = np.percentile(conf_in, 100-tpr)
    fp = np.sum(conf_out >=  perc)
    fpr = np.sum(conf_out >=  perc)/len(conf_out)
    return fpr, perc


def get_calib(pys, y_true, M=15):
    # Put the confidence into M bins
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    # In percent
    ECE, MCE = ECE*100, MCE*100

    return ECE, MCE


def get_nll(pys, y_true, averaged=True):
    pys = torch.from_numpy(pys).float()
    y_true = torch.from_numpy(y_true).long()
    dist = torch.distributions.Categorical(pys)
    nlls = -dist.log_prob(y_true)
    nll = nlls.mean() if averaged else nlls.sum()
    return nll.item()


def timing(fun):
    """
    Return the original output(s) and a wall-clock timing in second.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    ret = fun()
    end.record()

    torch.cuda.synchronize()

    return ret, start.elapsed_time(end)/1000
