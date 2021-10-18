import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy, MultitaskVariationalStrategy
import rgpr.kernel as rgp_kernel


class GPModel(ApproximateGP):

    def __init__(self, inducing_points):

        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPResidual(ApproximateGP):

    def __init__(self, base_model, inducing_points, num_classes, orig_shape, kernel):
        inducing_points = inducing_points.flatten(1)
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0), batch_shape=torch.Size([num_classes])
        )
        variational_strategy = MultitaskVariationalStrategy(
            VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=False
            ), num_tasks=num_classes
        )
        super().__init__(variational_strategy)

        self.base_model = base_model
        self.orig_shape = orig_shape
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel() if kernel == 'RBF' else rgp_kernel.DSCSKernel()
        )

    def forward(self, x):
        orig_shape = [x.shape[0]] + self.orig_shape

        with torch.no_grad():
            fx = self.base_model(x.reshape(orig_shape)).T

        mean_x = fx + self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
