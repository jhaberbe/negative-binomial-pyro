import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import ClippedAdam
import pyro.distributions as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model(y, size_factors, individual, group):
    # Move inputs to the appropriate device
    y = y.to(device)
    size_factors = size_factors.to(device)
    individual = individual.to(device)
    group = group.to(device)

    C = y.shape[0]              # number of cells
    N = group.shape[0]          # number of individuals
    G = int(group.max().item()) + 1

    alpha = pyro.sample("alpha", dist.Gamma(torch.ones(G, device=device) * 2,
                                            torch.ones(G, device=device)).to_event(1))
    beta  = pyro.sample("beta",  dist.Gamma(torch.ones(G, device=device) * 2,
                                            torch.ones(G, device=device)).to_event(1))
    r = pyro.sample("r", dist.Gamma(torch.tensor(2.0, device=device),
                                    torch.tensor(1.0, device=device)))

    with pyro.plate("individuals", N):
        a_i = alpha[group]
        b_i = beta[group]
        p_i = pyro.sample("p", dist.Beta(a_i, b_i))

    p_c = p_i[individual]
    mu = (r * (1 - p_c) / p_c) * size_factors

    with pyro.plate("cells", C):
        pyro.sample("obs", dist.NegativeBinomial(total_count=r,
                                                 logits=(mu / r).log()), obs=y)


def guide(y, size_factors, individual, group):
    # Move inputs to the appropriate device
    y = y.to(device)
    size_factors = size_factors.to(device)
    individual = individual.to(device)
    group = group.to(device)

    N = group.shape[0]
    G = int(group.max().item()) + 1

    # Group-level posterior on alpha, beta
    log_alpha_loc = pyro.param("log_alpha_loc", torch.zeros(G, device=device))
    log_alpha_scale = pyro.param("log_alpha_scale",
                                 torch.ones(G, device=device) * 0.1,
                                 constraint=dist.constraints.positive)
    log_beta_loc = pyro.param("log_beta_loc", torch.zeros(G, device=device))
    log_beta_scale = pyro.param("log_beta_scale",
                                torch.ones(G, device=device) * 0.1,
                                constraint=dist.constraints.positive)

    pyro.sample("alpha", dist.LogNormal(log_alpha_loc, log_alpha_scale).to_event(1))
    pyro.sample("beta",  dist.LogNormal(log_beta_loc, log_beta_scale).to_event(1))

    # Global r
    log_r_loc = pyro.param("log_r_loc", torch.tensor(0.0, device=device))
    log_r_scale = pyro.param("log_r_scale", torch.tensor(0.1, device=device),
                             constraint=dist.constraints.positive)
    pyro.sample("r", dist.LogNormal(log_r_loc, log_r_scale))

    # Per-individual p_i using logistic-normal
    logit_p_loc = pyro.param("logit_p_loc", torch.zeros(N, device=device))
    logit_p_scale = pyro.param("logit_p_scale",
                               torch.ones(N, device=device) * 0.1,
                               constraint=dist.constraints.positive)
    with pyro.plate("individuals", N):
        pyro.sample("p", dist.TransformedDistribution(
            dist.Normal(logit_p_loc, logit_p_scale),
            [dist.transforms.SigmoidTransform()]
        ))
