import torch
import os
import logging
import math
import matplotlib.pyplot as plt
import time
import sys

import gpytorch.settings as gpt_settings

from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import HigherOrderGP, SingleTaskGP, ModelList
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.optim.fit import fit_gpytorch_torch

from gpytorch.mlls import ExactMarginalLogLikelihood

SMOKE_TEST = os.environ.get("SMOKE_TEST")

filename = sys.argv[1]
regret_file = f"./outputs_dixonprice/regret_{filename}.txt"
runtime_file = f"./outputs_dixonprice/runtime_{filename}.txt"


torch.manual_seed(time.time())
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda:4")
dtype = torch.float

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


n_init = 50
beta = 1.0

if SMOKE_TEST:
    n_batches = 1
    batch_size = 2
    n_trials = 1
else:
    n_batches = 70
    batch_size = 1
    n_trials = 3

models_used = (
    "rnd",
    "ei",
    "ucb",
    # "ei_hogp_cf",
    "comp-ucb",
    "bomcf"
)


with gpt_settings.cholesky_jitter(1e-4):
    c_batched, objective, bounds, num_samples, global_minima = prepare_data()
    train_X_init = gen_rand_points(bounds, n_init)
    train_Y_init = c_batched(train_X_init)

    train_X = {k: train_X_init.clone() for k in models_used}
    train_Y = {k: train_Y_init.clone() for k in train_X}

    for i in range(n_batches):
        runtimes = {}

        # get best observations, log status
        best_f = {k: objective(v).max().detach() for k, v in train_Y.items()}

        print(
            f"It {i+1:>2}/{n_batches}, best obs.: "
            ", ".join([f"{k}: {v:.3f}" for k, v in best_f.items()])
        )

        # generate random candidates
        tic = time.monotonic()
        cands = {}
        cands["rnd"] = gen_rand_points(bounds, batch_size)
        runtimes["rnd"] = time.monotonic() - tic
        # hyperparameters for LBFGS
        optimize_acqf_kwargs = {
            "q": batch_size,
            "num_restarts": 50,
            "raw_samples": 1024,
            "dtype": torch.double
        }
        sampler = IIDNormalSampler(128)

        # Vanilla EI

        tic = time.monotonic()

        train_Y_ei = objective(train_Y["ei"]).unsqueeze(-1)
        model_ei = SingleTaskGP(
            train_X["ei"],
            train_Y_ei,
            input_transform=Normalize(train_X["ei"].shape[-1]),
            outcome_transform=Standardize(train_Y_ei.shape[-1]),
        )

        mll = ExactMarginalLogLikelihood(model_ei.likelihood, model_ei)
        fit_gpytorch_torch(
            mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})

        # generate qEI candidate (single output modeling)
        qEI = qExpectedImprovement(
            model_ei, best_f=best_f["ei"], sampler=sampler)
        try:
            cands["ei"] = optimize_ei(qEI, bounds, **optimize_acqf_kwargs)
        except:
            # if LBFGS doesn't converge, no new attempt
            cands["ei"] = None

        runtimes["ei"] = time.monotonic() - tic

        # Vanilla UCB

        tic = time.monotonic()

        train_Y_ucb = objective(train_Y["ucb"]).unsqueeze(-1)
        model_ucb = SingleTaskGP(
            train_X["ucb"],
            train_Y_ucb,
            input_transform=Normalize(train_X["ucb"].shape[-1]),
            outcome_transform=Standardize(train_Y_ucb.shape[-1]),
        )

        mll = ExactMarginalLogLikelihood(model_ucb.likelihood, model_ucb)
        fit_gpytorch_torch(
            mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})

        # generate qEI candidate (single output modeling)
        qUCB = qUpperConfidenceBound(model_ucb, beta=beta, sampler=sampler)
        try:
            cands["ucb"] = optimize_ucb(qUCB, bounds, **optimize_acqf_kwargs)
        except:
            cands["ucb"] = None

        runtimes["ucb"] = time.monotonic() - tic

        # Comp-UCB

        tic = time.monotonic()

        models_comp_ucb = []
        for itr in range(DixonPrice["d"]):
            gp = SingleTaskGP(
                train_X["comp-ucb"],
                train_Y["comp-ucb"][:, itr:itr+1],
                input_transform=Normalize(train_X["comp-ucb"].shape[-1]),
                outcome_transform=Standardize(
                    train_Y["comp-ucb"][:, itr:itr+1].shape[-1])
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_torch(
                mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})
            models_comp_ucb.append(gp)
        final_model_comp_ucb = ModelList(*models_comp_ucb)
        qUCB_comp_ucb = qUpperConfidenceBound(
            final_model_comp_ucb,
            beta=beta,
            sampler=sampler,
            objective=objective
        )
        try:
            cands["comp-ucb"] = optimize_ucb(qUCB_comp_ucb,
                                             bounds, **optimize_acqf_kwargs)
        except:
            cands["comp-ucb"] = None

        runtimes["comp-ucb"] = time.monotonic() - tic

        # BOMCF

        tic = time.monotonic()

        models_bomcf = []
        for itr in range(DixonPrice["d"]):
            gp = SingleTaskGP(
                train_X["bomcf"],
                train_Y["bomcf"][:, itr:itr+1],
                input_transform=Normalize(train_X["bomcf"].shape[-1]),
                outcome_transform=Standardize(
                    train_Y["bomcf"][:, itr:itr+1].shape[-1])
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_torch(
                mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})
            models_bomcf.append(gp)
        final_model_bomcf = ModelList(*models_bomcf)
        qEI_bomcf = qExpectedImprovement(
            final_model_bomcf,
            best_f=best_f["bomcf"],
            sampler=sampler,
            objective=objective
        )
        try:
            cands["bomcf"] = optimize_ei(
                qEI_bomcf, bounds, **optimize_acqf_kwargs)
        except:
            cands["bomcf"] = None

        runtimes["bomcf"] = time.monotonic() - tic

        # HOGP

        # tic = time.monotonic()

        # model_ei_hogp_cf = HigherOrderGP(
        #     train_X["ei_hogp_cf"],
        #     train_Y["ei_hogp_cf"],
        #     outcome_transform=FlattenedStandardize(train_Y["ei_hogp_cf"].shape[1:]),
        #     input_transform=Normalize(train_X["ei_hogp_cf"].shape[-1]),
        #     latent_init="gp",
        # )

        # mll = ExactMarginalLogLikelihood(model_ei_hogp_cf.likelihood, model_ei_hogp_cf)
        # fit_gpytorch_torch(mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})

        # # generate qEI candidate (multi-output modeling)
        # qEI_hogp_cf = qExpectedImprovement(
        #     model_ei_hogp_cf,
        #     best_f=best_f["ei_hogp_cf"],
        #     sampler=sampler,
        #     objective=objective,
        # )
        # cands["ei_hogp_cf"] = optimize_ei(qEI_hogp_cf, bounds, **optimize_acqf_kwargs)

        # runtimes["ei_hogp_cf"] = time.monotonic() - tic

        # make observations and update data
        regrets = {}

        for k, Xold in train_X.items():
            if cands[k] == None:
                continue
            Xnew = cands[k]
            if Xnew.shape[0] > 0:
                train_X[k] = torch.cat([Xold, Xnew])
                train_Y[k] = torch.cat([train_Y[k], c_batched(Xnew)])
                vals = -torch.sum(c_batched(Xnew), -1)
                regrets[k] = vals - global_minima
        beta = beta * (0.999 ** batch_size)

        # Log outputs
        # run times
        with open(runtime_file, "a+") as f:
            f.write(f"Iteration {i}\n")
            for method in models_used:
                f.write(f"{method} -- {runtimes[method]}\n")
            f.close()
        # regret
        with open(regret_file, "a+") as f:
            f.write(f"Iteration {i}\n")
            for method in models_used:
                if method in regrets:
                    f.write(f"{method} -- {regrets[method]}\n")
                else:
                    f.write(f"{method} -- None\n")
            f.close()

        # output
        print(f"{i}")
        print(f"Runtimes: {runtimes}")
        print(f"Regrets: {regrets}")
