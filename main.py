import sys
sys.path.append(".")

import src.model
import scanpy as sc

import torch

import pyro
from pyro.nn import PyroModule
from pyro.infer import SVI, Trace_ELBO, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
import pyro.distributions as dist

import argparse



def main():
    parser = argparse.ArgumentParser(description="Negative Binomial Model.")

    # Model arguments
    parser.add_argument("--adata", type=str, default="adata.h5ad", help="Path to AnnData object")
    parser.add_argument("--gene", type=int, default=0, help="gene_index")

    args = parser.parse_args()

    adata = sc.read_h5ad(args.adata)
    adata = adata[adata.obs["Cell.Subtype"].eq("Macrophage")].copy()
    gene = adata.var_names[args.gene]

    df = adata.obs \
        .assign(expression = adata[:, gene].X.todense().reshape(-1).tolist()[0]) \
        .assign(size_factor = adata.X.sum(axis=1).reshape(-1).tolist()[0]) \
        .eval("size_factor = size_factor / size_factor.mean()") \
        [["orig.ident", "cogdx", "expression", "size_factor"]]

    df["cogdx"].replace({1: 0, 2: 1, 4: 2}, inplace=True)

    y = torch.tensor(df["expression"]).float()
    size_factors = torch.tensor(df["size_factor"]).float()
    individual = torch.tensor(df["orig.ident"].cat.codes).long()
    group = torch.tensor(df["cogdx"]).long()

    pyro.clear_param_store()
    optimizer = ClippedAdam({'lr': 1e-3, 'clip_norm': 10.0})
    elbo = TraceMeanField_ELBO(num_particles=10)
    svi = SVI(src.model.model, src.model.guide, optimizer, loss=elbo)

    losses = []
    for step in range(5_000):
        loss = svi.step(
            y, 
            size_factors, 
            individual, 
            group
        )

        losses.append(loss)

        if step % 100 == 0:
            print(f"Step {step} - ELBO: {loss:.2f}")
    
    with open(f"output/macrophage/{gene.replace('/', '--')}.txt", "w") as f:
        f.write("Model parameters:\n")
        for name, param in pyro.get_param_store().items():
            f.write(f"{name}: {param.detach().clone().tolist()}\n")

if __name__ == "__main__":
    main()
