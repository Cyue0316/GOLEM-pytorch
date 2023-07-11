import torch
import numpy as np
import torch.nn as nn

class GolemModel(nn.Module):
    """Set up the objective function of GOLEM.
        Hyperparameters:
            (1) GOLEM-NV: equal_variances=False, lambda_1=2e-3, lambda_2=5.0.
            (2) GOLEM-EV: equal_variances=True, lambda_1=2e-2, lambda_2=5.0.
        """

    def __init__(self, n, d, lambda_1, lambda_2, equal_variances=True, seed=1, B_init=None):
        super(GolemModel, self).__init__()
        self.n = n
        self.d = d
        self.seed = seed
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.equal_variances = equal_variances
        self.B_init = B_init
        # Variables
        if self.B_init is not None:
            self.B = nn.Parameter(torch.tensor(self.B_init,requires_grad=True))
        else:
            self.B = nn.Parameter(torch.zeros(self.d, self.d, requires_grad=True))
        self.B = self._preprocess(self.B)

    def _preprocess(self,B):
        DiagB = torch.diag(B)
        b_diag = torch.diag_embed(DiagB)
        return nn.Parameter(B-b_diag)

    def _compute_likelihood(self, X):
        """Compute (negative log) likelihood in the linear Gaussian case.
        """
        if self.equal_variances:  # Assuming equal noise variances
            return 0.5 * self.d * torch.log(
                torch.square(
                    torch.norm(X - X @ self.B)
                )
            ) - torch.slogdet(torch.eye(self.d) - self.B)[1]

        # - torch.slogdet(1.0 * torch.eye(self.d) - self.B)[1] + self.d * np.log(1.0)
        else:  # Assuming non-equal noise variances
            return 0.5 * torch.sum(
                torch.log(
                    torch.sum(
                        torch.square(X - X @ self.B), dim=0
                    )
                )
            ) - torch.slogdet(torch.eye(self.d) - self.B)[1]

    def _compute_L1_penalty(self):
        return torch.norm(self.B, p=1)

    def _compute_h(self):
        return torch.trace(torch.matrix_exp(self.B * self.B)) - self.d

    def forward(self, X):
        # Likelihood, penalty terms and score
        self.likelihood = self._compute_likelihood(X)
        self.L1_penalty = self._compute_L1_penalty()
        self.h = self._compute_h()
        self.score = self.likelihood + self.lambda_1 * self.L1_penalty + self.lambda_2 * self.h
        return self.score, self.likelihood, self.h, self.B

    def __call__(self, x):
        return self.forward(x)


if __name__ == '__main__':
    # X = torch.ones([1000, 20])
    # # GOLEM-EV
    # golem = GolemModel(n=1000, d=20, lambda_1=2e-3, lambda_2=0.5, seed=1)
    # score, _, _, B = golem(X)
    # print(score)
    # print(B)
    batch_size = 32
    num_steps = 10
