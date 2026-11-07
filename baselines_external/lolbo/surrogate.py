"""LOL-BO's deep-kernel sparse GP surrogate, device-agnostic.

`GPModelDKL` is the vendored model (SVGP: Cholesky variational distribution over
learned inducing points, ConstantMean, ScaleKernel(RBF), preceded by a
`DenseNetwork` deep-kernel feature extractor with hidden dims (256, 256)).

`initialize_surrogate` and `update_surrogate` mirror `LOLBOState.initialize_surrogate_model`
(vendor/lolbo/lolbo.py) and `update_surr_model` (vendor/lolbo/utils/utils.py):

    likelihood      GaussianLikelihood
    inducing points first min(n, 1024) training latents
    objective       PredictiveLogLikelihood(num_data=n_train)
    optimiser       Adam, lr = learning_rte, grad-norm clip 1.0
    minibatch       min(n, 128), shuffled
    epochs          init_n_epochs (20) on all data the first time,
                    num_update_epochs (2) on the newest batch thereafter

The only edits are `.cuda()` -> `.to(device)` and returning the posterior helper
used for the promise-vs-delivery diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import gpytorch
import torch
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.mlls import PredictiveLogLikelihood
from torch.utils.data import DataLoader, TensorDataset

VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from lolbo.utils.bo_utils.ppgpr import GPModelDKL as _UpstreamGPModelDKL  # noqa: E402

__all__ = ["initialize_surrogate", "update_surrogate", "posterior_mean_std", "GPModelDKL"]


class GPModelDKL(_UpstreamGPModelDKL):
    """Upstream GPModelDKL with a BoTorch-current `posterior()`.

    The vendored `posterior()` calls `GPyTorchPosterior(mvn=dist)`; that keyword
    was renamed to `distribution` in BoTorch 0.9 (this repo runs 0.18), and
    BoTorch's `MaxPosteriorSampling` also now passes `posterior_transform`.
    The distribution being returned is unchanged: `likelihood(self(X))`, i.e.
    the same noisy posterior LOL-BO Thompson-samples from.
    """

    def posterior(
        self,
        X,
        output_indices=None,
        observation_noise=False,
        posterior_transform=None,
        *args,
        **kwargs,
    ) -> GPyTorchPosterior:
        self.eval()
        self.likelihood.eval()
        dist = self.likelihood(self(X))
        try:
            posterior = GPyTorchPosterior(distribution=dist)
        except TypeError:  # BoTorch < 0.9
            posterior = GPyTorchPosterior(mvn=dist)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


def initialize_surrogate(
    train_z: torch.Tensor,
    device: torch.device,
    n_inducing: int = 1024,
) -> Tuple[GPModelDKL, PredictiveLogLikelihood]:
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    n_pts = min(train_z.shape[0], n_inducing)
    model = GPModelDKL(train_z[:n_pts, :].to(device), likelihood=likelihood).to(device)
    mll = PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.size(-2))
    model = model.eval()
    return model, mll


def update_surrogate(
    model: GPModelDKL,
    mll: PredictiveLogLikelihood,
    learning_rate: float,
    train_z: torch.Tensor,
    train_y: torch.Tensor,
    n_epochs: int,
    device: torch.device,
) -> GPModelDKL:
    model = model.train()
    optimizer = torch.optim.Adam([{"params": model.parameters(), "lr": learning_rate}], lr=learning_rate)
    train_bsz = min(len(train_y), 128)
    dataset = TensorDataset(train_z.to(device), train_y.to(device))
    loader = DataLoader(dataset, batch_size=train_bsz, shuffle=True)
    for _ in range(n_epochs):
        for inputs, scores in loader:
            optimizer.zero_grad()
            output = model(inputs.to(device))
            loss = -mll(output, scores.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    model = model.eval()
    return model


@torch.no_grad()
def posterior_mean_std(model: GPModelDKL, z: torch.Tensor, device: torch.device):
    """Surrogate belief at proposal points, for the promise-vs-delivery log."""
    model.eval()
    posterior = model.posterior(z.to(device))
    mean = posterior.mean.squeeze(-1).detach().cpu()
    std = posterior.variance.clamp_min(1e-12).sqrt().squeeze(-1).detach().cpu()
    return mean, std
