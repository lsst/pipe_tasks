import gpytorch
import torch
import george
from george import kernels
import treegp

__all__ = ["GaussianProcessTreegp", "GaussianProcessHODLRSolver", "GaussianProcessGPyTorch"]


# Vanilla Gaussian Process regression using treegp package
# There is no fancy O(N*log(N)) solver here, just the basic GP regression (Cholesky).
class GaussianProcessTreegp:
    """
    Gaussian Process regression using treegp package.

    This class implements Gaussian Process regression using the treegp package. It provides methods for fitting the
    regression model and making predictions.

    Attributes:
        std (float): The standard deviation parameter for the Gaussian Process kernel.
        l (float): The correlation length parameter for the Gaussian Process kernel.
        white_noise (float): The white noise parameter for the Gaussian Process kernel.
        mean (float): The mean parameter for the Gaussian Process kernel.

    Methods:
        fit(x_good, y_good): Fits the Gaussian Process regression model to the given training data.
        predict(x_bad): Makes predictions using the fitted Gaussian Process regression model.

    """

    def __init__(self, std=1.0, correlation_length=1.0, white_noise=0.0, mean=0.0):
        """
        Initializes a new instance of the gp_treegp class.

        Args:
            std (float, optional): The standard deviation parameter for the Gaussian Process kernel. Defaults to 2.
            correlation_length (float, optional): The correlation length parameter for the Gaussian Process kernel.
                Defaults to 1.
            white_noise (float, optional): The white noise parameter for the Gaussian Process kernel. Defaults to 0.
            mean (float, optional): The mean parameter for the Gaussian Process kernel. Defaults to 0.

        """
        self.std = std
        self.l = correlation_length
        self.white_noise = white_noise
        self.mean = mean

    def fit(self, x_good, y_good):
        """
        Fits the Gaussian Process regression model to the given training data.

        Args:
            x_good (array-like): The input features of the training data.
            y_good (array-like): The target values of the training data.

        """
        KERNEL = "%.2f**2 * RBF(%f)" % ((self.std, self.l))
        self.gp = treegp.GPInterpolation(
            kernel=KERNEL,
            optimizer="none",
            normalize=True,
            p0=[3000.0, 0.0, 0.0],
            white_noise=self.white_noise,
        )
        self.gp.initialize(x_good, y_good)
        self.gp.solve()

    def predict(self, x_bad):
        """
        Makes predictions using the fitted Gaussian Process regression model.

        Args:
            x (array-like): The input features for which to make predictions.

        Returns:
            array-like: The predicted target values.

        """
        y_pred = self.gp.predict(x_bad)
        return y_pred


# GP using HOLDR solver from george package
class GaussianProcessHODLRSolver:
    """
    A class representing a Gaussian Process solver using the HODLR solver method.

    Parameters:
    - std (float): The standard deviation of the Gaussian process.
    - correlation_length (float): The correlation length of the Gaussian process.
    - white_noise (float): The white noise level of the Gaussian process.
    - mean (float): The mean value of the Gaussian process.

    Methods:
    - fit(x_good, y_good, **kwargs): Fits the Gaussian process to the given training data.
    - predict(x_bad): Predicts the values of the Gaussian process at the given test points.

    """

    def __init__(self, std=1.0, correlation_length=1.0, white_noise=0.0, mean=0.0):
        """
        Initializes a GaussianProcessHODLRSolver object.

        Parameters:
        - std (float): The standard deviation of the Gaussian process.
        - correlation_length (float): The correlation length of the Gaussian process.
        - white_noise (float): The white noise level of the Gaussian process.
        - mean (float): The mean value of the Gaussian process.
        """

        self.variance = std**2
        self.correlation_length = correlation_length
        self.white_noise = white_noise
        self.mean = mean

    def fit(self, x_good, y_good, **kwargs):
        """
        Fits the Gaussian process to the given training data.

        Parameters:
        - x_good (array-like): The input training data.
        - y_good (array-like): The target training data.
        - **kwargs: Additional keyword arguments to be passed to the GP solver.

        """
        kernel = self.variance * kernels.ExpSquaredKernel(self.correlation_length, ndim=2)
        self.gp = george.GP(
            kernel, mean=self.mean, fit_kernel=False, solver=george.HODLRSolver, seed=42, **kwargs
        )
        self.gp.compute(x_good, yerr=self.white_noise)
        self._y_good = y_good

    def predict(self, x_bad):
        """
        Predicts the values of the Gaussian process at the given test points.

        Parameters:
        - x_bad (array-like): The test points at which to predict the Gaussian process values.

        Returns:
        - y_pred (array-like): The predicted values of the Gaussian process at the test points.

        """
        y_pred = self.gp.predict(self._y_good, x_bad, return_var=False, return_cov=False)
        return y_pred


# GP using gpytorch package
# This class is a wrapper around the gpytorch.models.ExactGP class.
# This is where the GP model is defined.
class GPRegressionModelEXACT(gpytorch.models.ExactGP):
    """
    Gaussian Process regression model using the ExactGP framework.

    Args:
        x_good (torch.Tensor): The input training data.
        y_good (torch.Tensor): The target training data.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.

    Attributes:
        mean_module (gpytorch.means.Mean): The mean module for the GP model.
        covar_module (gpytorch.kernels.Kernel): The covariance module for the GP model.
    """

    def __init__(self, x_good, y_good, likelihood):
        super(GPRegressionModelEXACT, self).__init__(x_good, y_good, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        """
        Forward pass of the GP regression model.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            gpytorch.distributions.MultivariateNormal: The predicted distribution over the output data.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcessGPyTorch:
    """
    A class representing a Gaussian Process regression model using GPyTorch.

    Parameters:
    - std (float): The standard deviation of the output.
    - correlation_length (float): The length scale of the covariance function.
    - white_noise (float): The noise level of the likelihood function.
    - mean (float): The mean of the output.

    Methods:
    - fit(x_good, y_good): Fits the Gaussian Process model to the given training data.
    - predict(x): Predicts the output for the given input using the trained model.

    """

    def __init__(self, std=1.0, correlation_length=1.0, white_noise=1.0, mean=0.0):
        """
        Initializes the GaussianProcessGPyTorch object.

        Parameters:
        - std (float): The standard deviation of the output.
        - correlation_length (float): The length scale of the covariance function.
        - white_noise (float): The noise level of the likelihood function.
        - mean (float): The mean of the output.
        """
        self.hypers = {
            "likelihood.noise_covar.noise": torch.tensor(white_noise, dtype=torch.float32),
            "covar_module.base_kernel.lengthscale": torch.tensor(correlation_length, dtype=torch.float32),
            "covar_module.outputscale": torch.tensor(std, dtype=torch.float32),
            "mean_module.constant": torch.tensor(mean, dtype=torch.float32),
        }

        if torch.cuda.is_available():
            self.cuda = True
        else:
            self.cuda = False

        if self.cuda:
            self.hypers = {k: v.cuda() for k, v in self.hypers.items()}

    def fit(self, x_good, y_good):
        """
        Fits the Gaussian Process model to the given training data.

        Parameters:
        - x_good (list or numpy array): The input training data.
        - y_good (list or numpy array): The output training data.
        """
        x = torch.tensor(x_good, dtype=torch.float32)
        y = torch.tensor(y_good, dtype=torch.float32)

        if self.cuda:
            (
                x,
                y,
            ) = (
                x.cuda(),
                y.cuda(),
            )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPRegressionModelEXACT(x, y, self.likelihood)
        if self.cuda:
            model = model.cuda()
            likelihood = likelihood.cuda()
        self.model.initialize(**self.hypers)
        self.model.eval()
        self.likelihood.eval()

    def predict(self, x):
        """
        Predicts the output for the given input using the trained model.

        Parameters:
        - x (list or numpy array): The input data for prediction.

        Returns:
        - y_predict (numpy array): The predicted output for the given input.
        """
        x = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            prediction = self.likelihood(self.model(x))
            y_predict = prediction.mean
        return y_predict.numpy()
