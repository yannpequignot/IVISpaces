# from https://github.com/AntixK/Spectral-Stein-Gradient
import torch.tensor as Tensor
import torch
from abc import abstractmethod


class BaseScoreEstimator:

    @staticmethod
    def rbf_kernel(x1: Tensor,
                   x2: Tensor,
                   sigma: float) -> Tensor:
        return torch.exp(-((x1 - x2).pow(2).sum(-1))/(2 * sigma ** 2))

    def gram_matrix(self,
             x1: Tensor,
             x2: Tensor,
             sigma: float) -> Tensor:
        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor
        return self.rbf_kernel(x1, x2, sigma)

    def grad_gram(self,
                  x1: Tensor,
                  x2: Tensor,
                  sigma: float) -> Tensor:
        """
        Computes the gradients of the RBF gram matrix with respect
        to the inputs x1 an x2. It is given by
        .. math::
            \nabla_x1 k(x1, x2) = k(x1, x2) \frac{x1- x2}{\sigma^2}
            \nabla_x2 k(x1, x2) = k(x1, x2) -\frac{x1- x2}{\sigma^2}
        :param x1: (Tensor) [N x D]
        :param x2: (Tensor) [M x D]
        :param sigma: (Float) Width of the RBF kernel
        :return: Gram matrix [N x M],
                 gradients with respect to x1 [N x M x D],
                 gradients with respect to x2 [N x M x D]
        """
        with torch.no_grad():
            Kxx = self.gram_matrix(x1, x2, sigma)

            x1 = x1.unsqueeze(-2)  # Make it into a column tensor
            x2 = x2.unsqueeze(-3)  # Make it into a row tensor
            diff = (x1 - x2) / (sigma ** 2)

            dKxx_dx1 = Kxx.unsqueeze(-1) * (-diff)
            dKxx_dx2 = Kxx.unsqueeze(-1) * diff
            return Kxx, dKxx_dx1, dKxx_dx2

    def heuristic_sigma(self,
                        x:Tensor,
                        xm: Tensor) -> Tensor:
        """
        Uses the median-heuristic for selecting the
        appropriate sigma for the RBF kernel based
        on the given samples.
        The kernel width is set to the media of the
        pairwise distances between x and xm.
        :param x: (Tensor) [N x D]
        :param xm: (Tensor) [M x D]
        :return:
        """

        with torch.no_grad():
            x1 = x.unsqueeze(-2)   # Make it into a column tensor
            x2 = xm.unsqueeze(-3)  # Make it into a row tensor

            pdist_mat = torch.sqrt(((x1 - x2) ** 2).sum(dim = -1)) # [N x M]
            kernel_width = torch.median(torch.flatten(pdist_mat))
            return kernel_width

    @abstractmethod
    def compute_score_gradients(self, x: Tensor, xm: Tensor = None):
        raise NotImplementedError

    def __call__(self, x: Tensor, xm: Tensor = None):
        return self.compute_score_gradients(x, xm)
    
class SpectralSteinEstimator(BaseScoreEstimator):
    def __init__(self,
                 eta: float = None,
                 num_eigs: int = None) -> None:
        self.eta = eta
        self.num_eigs = num_eigs

    def nystrom_method(self,
                       x: Tensor,
                       eval_points: Tensor,
                       eigen_vecs: Tensor,
                       eigen_vals: Tensor,
                       kernel_sigma: float) -> Tensor:
        """
        Implements the Nystrom method for approximating the
        eigenfunction (generalized eigenvectors) for the kernel
        at x using the M eval_points (x_m). It is given
        by -
         .. math::
            phi_j(x) = \frac{M}{\lambda_j} \sum_{m=1}^M u_{jm} k(x, x_m)
        :param x: (Tensor) Point at which the eigenfunction is evaluated [N x D]
        :param eval_points: (Tensor) Sample points from the data of ize M [M x D]
        :param eigen_vecs: (Tensor) Eigenvectors of the gram matrix [M x M]
        :param eigen_vals: (Tensor) Eigenvalues of the gram matrix [M x 2]
        :param kernel_sigma: (Float) Kernel width
        :return: Eigenfunction at x [N x M]
        """
        M = torch.tensor(eval_points.size(-2), dtype=torch.float)
        Kxxm = self.gram_matrix(x, eval_points, kernel_sigma)
        phi_x =  torch.sqrt(M) * Kxxm @ eigen_vecs

        phi_x *= 1. / eigen_vals[:,0] # Take only the real part of the eigenvals
                                      # as the Im is 0 (Symmetric matrix)
        return phi_x


    def compute_score_gradients(self,
                                x: Tensor,
                                xm: Tensor = None) -> Tensor:
        """
        Computes the Spectral Stein Gradient Estimate (SSGE) for the
        score function. The SSGE is given by
        .. math::
            \nabla_{xi} phi_j(x) = \frac{1}{\mu_j M} \sum_{m=1}^M \nabla_{xi}k(x,x^m) \phi_j(x^m)
            \beta_{ij} = -\frac{1}{M} \sum_{m=1}^M \nabla_{xi} phi_j (x^m)
            \g_i(x) = \sum_{j=1}^J \beta_{ij} \phi_j(x)
        :param x: (Tensor) Point at which the gradient is evaluated [N x D]
        :param xm: (Tensor) Samples for the kernel [M x D]
        :return: gradient estimate [N x D]
        """
        if xm is None:
            xm = x
            sigma = self.heuristic_sigma(xm, xm)
        else:
            # Account for the new data points too
            _xm = torch.cat((x, xm), dim=-2)
            sigma = self.heuristic_sigma(_xm, _xm)

        M = torch.tensor(xm.size(-2), dtype=torch.float)

        Kxx, dKxx_dx, _ = self.grad_gram(xm, xm, sigma)

        # Kxx = Kxx + eta * I
        if self.eta is not None:
            Kxx += self.eta * torch.eye(xm.size(-2)).to(xm.device)

        eigen_vals, eigen_vecs = torch.eig(Kxx, eigenvectors=True)

        if self.num_eigs is not None:
            eigen_vals = eigen_vals[:self.num_eigs]
            eigen_vecs = eigen_vecs[:, :self.num_eigs]

        phi_x = self.nystrom_method(x, xm, eigen_vecs, eigen_vals, sigma) # [N x M]

        # Compute the Monte Carlo estimate of the gradient of
        # the eigenfunction at x
        dKxx_dx_avg = dKxx_dx.mean(dim=-3) #[M x D]

        beta = - torch.sqrt(M) * eigen_vecs.t() @ dKxx_dx_avg
        beta *= (1. / eigen_vals[:, 0].unsqueeze(-1))

        # assert beta.allclose(beta1), f"incorrect computation {beta - beta1}"
        g = phi_x @ beta # [N x D]
        return g