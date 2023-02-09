import torch
import math
import os
import logging
import sys
import time
import multiprocessing

import gpytorch.settings as gpt_settings

from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import HigherOrderGP
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.optim.fit import fit_gpytorch_torch

from gpytorch.mlls import ExactMarginalLogLikelihood


filename = sys.argv[1]
regret_file = f"./outputs_dixonprice/hogp_regret_{filename}.txt"
runtime_file = f"./outputs_dixonprice/hogp_runtime_{filename}.txt"


SMOKE_TEST = os.environ.get("SMOKE_TEST")

torch.manual_seed(time.time())
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda:4")
dtype = torch.float

print("Using ", device)

DixonPrice = {
    "d": 5,
    "range_upper": 2.0,
    "range_lower": -2.0
}


def fi(x, i):
    if i == 0:
        return (x[0] - 1) ** 2
    else:
        return i * ((2 * (x[i] ** 2) - x[i - 1]) ** 2)


def env_cfun(x):
    return torch.tensor([fi(x, i) for i in range(DixonPrice["d"])])


def gen_rand_points(bounds, num_samples):
    points_nlzd = torch.rand(num_samples, bounds.shape[-1]).to(bounds)
    return bounds[0] + (bounds[1] - bounds[0]) * points_nlzd


def optimize_ei(qEI, bounds, **options):
    with gpt_settings.fast_computations(covar_root_decomposition=False):
        cands_nlzd, _ = optimize_acqf(
            qEI, bounds, **options,
        )
    return cands_nlzd


def optimize_ucb(qUCB, bounds, **options):
    with gpt_settings.fast_computations(covar_root_decomposition=False):
        cands_nlzd, _ = optimize_acqf(
            qUCB, bounds, **options,
        )
    return cands_nlzd


def prepare_data(device=device, dtype=dtype):
    bounds = torch.tensor(
        [[-2.0 for _ in range(DixonPrice["d"])],
         [2.0 for _ in range(DixonPrice["d"])]],
        device=device,
        dtype=dtype,
    )

    X0 = torch.tensor(
        [math.pow(2, -(math.pow(2, i + 1) - 2) / math.pow(2, i + 1))
         for i in range(DixonPrice["d"])],
        device=device, dtype=dtype
    )

    def c_batched(X):
        return torch.stack([env_cfun(x) for x in X])

    c_true = env_cfun(X0)
    global_minima = torch.sum(c_true)
    print(f"Global minima -- {global_minima} at {X0}")

    def neq_sum_quared_diff(samples):
        vals = torch.sum(samples, -1).square().mul(-1.0)
        return vals

    objective = GenericMCObjective(neq_sum_quared_diff)
    num_samples = 32

    return c_batched, objective, bounds, num_samples, global_minima

n_init = 10

if SMOKE_TEST:
    n_batches = 1
    batch_size = 2
    n_trials = 1
else:
    n_batches = 70
    batch_size = 1
    n_trials = 3

models_used = (
    "ei_hogp_cf",
)

m = multiprocessing.Manager()

all_objective_vals = []

with gpt_settings.cholesky_jitter(1e-4):
    c_batched, objective, bounds, num_samples, global_minima = prepare_data()
    train_X_init = gen_rand_points(bounds, n_init)
    train_Y_init = c_batched(train_X_init)

    # these will keep track of the points explored
    train_X = m.dict({k: train_X_init.clone() for k in models_used})
    train_Y = m.dict({k: train_Y_init.clone() for k in train_X})

    # run the BO loop
    for i in range(n_batches):
        with open(regret_file, "a+") as f:
            f.write(f"Iteration {i}\n")
        with open(runtime_file, "a+") as f:
            f.write(f"Iteration {i}\n")

        # get best observations, log status
        best_f = {k: objective(v).max().detach() for k, v in train_Y.items()}

        print(
            f"It {i+1:>2}/{n_batches}, best obs.: "
            ", ".join([f"{k}: {v:.3f}" for k, v in best_f.items()])
        )

        optimize_acqf_kwargs = {
            "q": batch_size,
            "num_restarts": 50,
            "raw_samples": 1024,
            "dtype": torch.double
        }
        sampler = IIDNormalSampler(128)

        def hogp(train_X, train_Y, best_f):
            tic = time.monotonic()
            model_ei_hogp_cf = HigherOrderGP(
                train_X["ei_hogp_cf"],
                train_Y["ei_hogp_cf"],
                outcome_transform=FlattenedStandardize(
                    train_Y["ei_hogp_cf"].shape[1:]),
                input_transform=Normalize(train_X["ei_hogp_cf"].shape[-1]),
                latent_init="gp",
            )

            mll = ExactMarginalLogLikelihood(
                model_ei_hogp_cf.likelihood, model_ei_hogp_cf)
            fit_gpytorch_torch(
                mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})

            # generate qEI candidate (multi-output modeling)
            qEI_hogp_cf = qExpectedImprovement(
                model_ei_hogp_cf,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
            )
            cands = optimize_ei(qEI_hogp_cf, bounds, **optimize_acqf_kwargs)

            if cands == None:
                return
            Xnew = cands
            if Xnew.shape[0] > 0:
                here = c_batched(Xnew)
                train_X["ei_hogp_cf"] = torch.cat(
                    [train_X["ei_hogp_cf"], Xnew])
                train_Y["ei_hogp_cf"] = torch.cat(
                    [train_Y["ei_hogp_cf"], here])
                vals = objective(here)
                print("hogp objective: ", vals)
                val = -torch.sum(here, -1)
                print("hogp regret: ", val)
                with open(regret_file, "a+") as f:
                    f.write(f"HOGP -- {val[0] - global_minima}\n")
            with open(runtime_file, "a+") as f:
                f.write(f"HOGP -- {time.monotonic() - tic}\n")

        p = multiprocessing.Process(target=lambda: hogp(train_X,
                                                        train_Y,
                                                        best_f["ei_hogp_cf"]))
        p.start()
        p.join(120)
        if p.is_alive():
            p.kill()
            print("killed after 120sec")
            p.join()
            with open(regret_file, "a+") as f:
                f.write("HOGP -- failed\n")
            with open(runtime_file, "a+") as f:
                f.write("HOGP -- 120.0\n")
