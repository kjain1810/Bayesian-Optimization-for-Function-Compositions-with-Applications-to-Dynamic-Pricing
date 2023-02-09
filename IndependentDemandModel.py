import torch
import time
import sys
import os
import random
import math
import multiprocessing

import gpytorch.settings as gpt_settings
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import qUpperConfidenceBound, qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP, ModelList, HigherOrderGP
from botorch.models.higher_order_gp import FlattenedStandardize
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.optim.fit import fit_gpytorch_torch

from scipy.optimize import minimize_scalar

SMOKE_TEST = os.environ.get("SMOKE_TEST")

filename = sys.argv[1]
regret_file = f"./outputs_pricing_3/regret_{filename}.txt"
runtime_file = f"./outputs_pricing_3/runtime_{filename}.txt"


torch.manual_seed(time.time())
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda:4")
dtype = torch.float

print("Using ", device)

z1 = random.uniform(1.0, 2.0)
z2 = random.uniform(-1.0, 1.0)
z3 = random.uniform(1.0, 2.0)
z4 = random.uniform(-1.0, 1.0)
z5 = random.uniform(2.0/3.0, 3.0/4.0)
z6 = random.uniform(0.75, 1.0)
z7 = random.uniform(2.0/3.0, 0.75)
z8 = random.uniform(0.75, 1.0)


def d1(x):
    return math.exp(-z1 * x - z2) / (1 + math.exp(-z1 * x - z2))


def d2(x):
    return math.exp(-z3 * x - z4) / (1 + math.exp(-z3 * x - z4))


def d3(x):
    return z5 - z6 * x


def d4(x):
    return z7 - z8 * x


min_d1 = minimize_scalar(lambda x: -x * d1(x), bounds=[0.5, 2.0])
min_d2 = minimize_scalar(lambda x: -x * d2(x), bounds=[0.5, 2.0])
min_d3 = minimize_scalar(lambda x: -x * d3(x), bounds=[1.0/3.0, 0.5])
min_d4 = minimize_scalar(lambda x: -x * d4(x), bounds=[1.0/3.0, 0.5])

print(
    f"best p1 = {min_d1.x} with revenue = {min_d1.x * d1(min_d1.x)} at z1={z1} and z2={z2}")
print(
    f"best p2 = {min_d2.x} with revenue = {min_d2.x * d2(min_d2.x)} at z3={z3} and z4={z4}")
print(
    f"best p3 = {min_d3.x} with revenue = {min_d3.x * d3(min_d3.x)} at z5={z5} and z6={z6}")
print(
    f"best p4 = {min_d4.x} with revenue = {min_d4.x * d4(min_d4.x)} at z7={z7} and z8={z8}")


def env_cfun(x):
    return torch.cat([torch.tensor([d1(x[0]), d2(x[1]), d3(x[2]), d4(x[3])]), x])


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
        [[0.5, 0.5, 1.0/3.0, 1.0/3.0], [2.0, 2.0, 0.5, 0.5]],
        device=device,
        dtype=dtype
    )

    def c_batched(X):
        return torch.stack([env_cfun(x) for x in X])

    global_maxima = d1(min_d1.x) + d2(min_d2.x) + d3(min_d3.x) + d4(min_d4.x)
    print(
        f"Global maxima -- {global_maxima} at ({(min_d1.x, min_d2.x, min_d3.x, min_d4.x)})")

    def neq_sum_quared_diff(samples):
        # print(samples.shape)
        here = torch.mul(samples[..., 0], samples[..., 4]) + \
            torch.mul(samples[..., 1], samples[..., 5]) + \
            torch.mul(samples[..., 2], samples[..., 6]) + \
            torch.mul(samples[..., 3], samples[..., 7])
        return here.sub(global_maxima).square().mul(-1.0)

    objective = GenericMCObjective(neq_sum_quared_diff)

    return c_batched, objective, bounds, global_maxima


n_init = 20
beta = 1.0
alpha = 0.9

n_batches = 70
batch_size = 1
n_trials = 3

models_used = (
    "rnd",
    "ei",
    "ucb",
    "comp_ucb",
    "ei_hogp_cf",
    "bomcf"
)

m = multiprocessing.Manager()

with gpt_settings.cholesky_jitter(1e-4):
    c_batched, objective, bounds, global_maxima = prepare_data()
    train_X_init = gen_rand_points(bounds, n_init)
    train_Y_init = c_batched(train_X_init)

    # these will keep track of the points explored
    train_X = m.dict({k: train_X_init.clone() for k in models_used})
    train_Y = m.dict({k: train_Y_init.clone() for k in train_X})

    # run the BO loop
    for itr in range(n_batches):
        # get best observations, log status
        best_f = {k: objective(v).max().detach() for k, v in train_Y.items()}

        optimize_acqf_kwargs = {
            "q": batch_size,
            "num_restarts": 50,
            "raw_samples": 1024,
            "dtype": torch.double,
        }
        sampler = IIDNormalSampler(128)

        def vanilla_EI(train_X, train_Y, best_f, cands, runtimes):
            print("\033[1;32m Doing Vanilla EI\033[0m")
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
                model_ei, best_f=best_f, sampler=sampler)
            try:
                cands["ei"] = optimize_ei(qEI, bounds, **optimize_acqf_kwargs)
            except:
                cands["ei"] = None

            runtimes["ei"] = time.monotonic() - tic

        def vanilla_UCB(train_X, train_Y, beta, cands, runtimes):
            print("\033[1;32m Doing Vanilla UCB\033[0m")
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
                cands["ucb"] = optimize_ucb(
                    qUCB, bounds, **optimize_acqf_kwargs)
            except:
                cands["ucb"] = None

            runtimes["ucb"] = time.monotonic() - tic

        def comp_ucb(train_X, train_Y, beta, cands, runtimes):
            print("\033[1;32m Doing Comp UCB\033[0m")
            tic = time.monotonic()

            models_comp_ucb = []
            for i in range(8):
                gp = SingleTaskGP(
                    train_X["comp_ucb"],
                    train_Y["comp_ucb"][:, i:i+1],
                    input_transform=Normalize(train_X["comp_ucb"].shape[-1]),
                    outcome_transform=Standardize(
                        train_Y["comp_ucb"][:, i:i+1].shape[-1])
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
                cands["comp_ucb"] = optimize_ucb(qUCB_comp_ucb,
                                                 bounds, **optimize_acqf_kwargs)
            except:
                cands["comp_ucb"] = None

            runtimes["comp_ucb"] = time.monotonic() - tic

        def bomcf(train_X, train_Y, best_f, cands, runtimes):
            print("\033[1;32m Doing bomcf\033[0m")
            tic = time.monotonic()

            models_bomcf = []
            for i in range(8):
                gp = SingleTaskGP(
                    train_X["bomcf"],
                    train_Y["bomcf"][:, i:i+1],
                    input_transform=Normalize(train_X["bomcf"].shape[-1]),
                    outcome_transform=Standardize(
                        train_Y["bomcf"][:, i:i+1].shape[-1])
                )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_torch(
                    mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})
                models_bomcf.append(gp)
            final_model_bomcf = ModelList(*models_bomcf)
            qEI_bomcf = qExpectedImprovement(
                final_model_bomcf,
                best_f=best_f,
                sampler=sampler,
                objective=objective
            )
            try:
                cands["bomcf"] = optimize_ei(
                    qEI_bomcf, bounds, **optimize_acqf_kwargs)
            except:
                cands["bomcf"] = None

            runtimes["bomcf"] = time.monotonic() - tic

        def hogp(train_X, train_Y, best_f, cands, runtimes):
            print("\033[1;32m Doing HOGP\033[0m")
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
            try:
                cands["ei_hogp_cf"] = optimize_ei(
                    qEI_hogp_cf, bounds, **optimize_acqf_kwargs)
            except:
                cands["ei_hogp_cf"] = None

            runtimes["ei_hogp_cf"] = time.monotonic() - tic
        cands = m.dict({})
        runtimes = m.dict({})

        # do random
        tic = time.monotonic()
        cands["rnd"] = gen_rand_points(bounds, batch_size)
        runtimes["rnd"] = time.monotonic() - tic

        # do Vanilla EI
        p = multiprocessing.Process(target=lambda: vanilla_EI(
            train_X, train_Y, best_f["ei"], cands, runtimes))
        p.start()
        p.join(20)
        if p.is_alive():
            p.kill()
            p.join()
            print("\033[0;31m killed ei after 20sec\033[0m")
            cands["ei"] = None
            runtimes["ei"] = 20.0
        print(cands)

        # do Vanilla UCB
        p = multiprocessing.Process(target=lambda: vanilla_UCB(
            train_X, train_Y, beta, cands, runtimes))
        p.start()
        p.join(20)
        if p.is_alive():
            p.kill()
            p.join()
            print("\033[0;31m killed ucb after 20sec\033[0m")
            cands["ucb"] = None
            runtimes["ucb"] = 20.0
        print(cands)

        # do hogp
        p = multiprocessing.Process(target=lambda: hogp(
            train_X, train_Y, best_f["ei_hogp_cf"], cands, runtimes))
        p.start()
        p.join(120)
        if p.is_alive():
            p.kill()
            p.join()
            print("\033[0;31m killed hogp after 120 sec\033[0m")
            cands["hogp"] = None
            runtimes["hogp"] = 120.0

        # do comp_ucb
        p = multiprocessing.Process(target=lambda: comp_ucb(
            train_X, train_Y, beta, cands, runtimes))
        p.start()
        p.join(40)
        if p.is_alive():
            p.kill()
            p.join()
            print("\033[0;31m killed comp_ucb after 40sec\033[0m")
            cands["comp_ucb"] = None
            runtimes["comp_ucb"] = 40.0

        # do bomcf
        p = multiprocessing.Process(target=lambda: bomcf(
            train_X, train_Y, best_f["bomcf"], cands, runtimes))
        p.start()
        p.join(40)
        if p.is_alive():
            p.kill()
            p.join()
            print("\033[0;31m killed bomcf after 40sec\033[0m")
            cands["bomcf"] = None
            runtimes["bomcf"] = 40.0

        # make observatios and update data
        regrets = {}
        for k, Xold in train_X.items():
            if cands[k] == None:
                continue
            Xnew = cands[k]
            if Xnew.shape[0] > 0:
                train_X[k] = torch.cat([Xold, Xnew])
                here = c_batched(Xnew)
                train_Y[k] = torch.cat([train_Y[k], here])
                # val = here[0][0] * here[0][2] + here[0][1] * here[0][3]
                val = here[0][0] * here[0][4] + here[0][1] * here[0][5] + \
                    here[0][2] * here[0][6] + here[0][3] * here[0][7]
                regrets[k] = global_maxima - val
        beta = beta * (alpha ** batch_size)
        print(train_X)
        print(train_Y)
        # Log outputs
        # run times
        with open(runtime_file, "a+") as f:
            f.write(f"Iteration {itr}\n")
            for method in models_used:
                f.write(f"{method} -- {runtimes[method]}\n")
            f.close()
        # regret
        with open(regret_file, "a+") as f:
            f.write(f"Iteration {itr}\n")
            for method in models_used:
                if method in regrets:
                    f.write(f"{method} -- {regrets[method]}\n")
                else:
                    f.write(f"{method} -- None\n")
            f.close()

        # output
        print(f"{itr}")
        print(f"Runtimes: {runtimes}")
        print(f"Regrets: {regrets}")
