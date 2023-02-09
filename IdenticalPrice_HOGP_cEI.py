import torch
import time
import sys
import os
import multiprocessing

import gpytorch.settings as gpt_settings
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models import SingleTaskGP, ModelList, HigherOrderGP
from botorch.models.transforms import Normalize, Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.optim.fit import fit_gpytorch_torch
from botorch.models.higher_order_gp import FlattenedStandardize

SMOKE_TEST = os.environ.get("SMOKE_TEST")

filename = sys.argv[1]
regret_file = f"./outputs_pricing_p0d0p1d1/hogp_regret_{filename}.txt"
runtime_file = f"./outputs_pricing_p0d0p1d1/hogp_runtime_{filename}.txt"


torch.manual_seed(time.time())
device = torch.device(
    "cpu") if not torch.cuda.is_available() else torch.device("cuda:4")
dtype = torch.float

print("Using ", device)

NUM_CUSTOMERS = 1000


def d0(p0):
    # exponential distribution
    return (torch.exp(-5 * p0)) * NUM_CUSTOMERS


def d1(p1):
    # gamma distribution
    return torch.abs(torch.exp(-p1 / 10) * (p1 - 1) / 10) * NUM_CUSTOMERS


def env_cfun(p):
    return torch.cat([torch.tensor([d0(p[0]), d1(p[1])]), p])


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
    # X = [M, D, L, tau]
    bounds = torch.tensor(
        [[0.0, 0.0], [1.0, 1.0]],
        device=device,
        dtype=dtype,
    )

    def c_batched(X, k=None):
        return torch.stack([env_cfun(x) for x in X])
    global_maxima = 97.372
    print(f"Global maxima -- {global_maxima}")

    def neq_sum_quared_diff(samples):
        return (torch.mul(samples[..., 0], samples[..., 2]) + torch.mul(samples[..., 1], samples[..., 3]))\
            .sub(global_maxima).square().mul(-1.0)

    objective = GenericMCObjective(neq_sum_quared_diff)
    num_samples = 32

    return c_batched, objective, bounds, num_samples, global_maxima


n_init = 5
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
    # "rnd",
    # "ei",
    # "ucb",
    # "comp-ucb",
    "ei_hogp_cf",
    "bomcf"
)


m = multiprocessing.Manager()

with gpt_settings.cholesky_jitter(1e-4):
    c_batched, objective, bounds, num_samples, global_maxima = prepare_data()
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
                val = here[0][0] * here[0][2] + here[0][1] * here[0][3]
                print("hogp regret: ", global_maxima - val)
                with open(regret_file, "a+") as f:
                    f.write(f"HOGP -- {global_maxima - val}\n")
            else:
                with open(regret_file, "a+") as f:
                    f.write(f"HOGP -- None\n")
            with open(runtime_file, "a+") as f:
                f.write(f"HOGP -- {time.monotonic() - tic}\n")

        def bomcf(train_X, train_Y, best_f):
            tic = time.monotonic()
            models_bomcf = []
            for i in range(4):
                gp = SingleTaskGP(
                        train_X["bomcf"],
                        train_Y["bomcf"][:, i:i+1],
                        input_transform=Normalize(train_X["bomcf"].shape[-1]),
                        outcome_transform=Standardize(train_Y["bomcf"][:, i:i+1].shape[-1])
                        )
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_torch(mll, options={"lr": 0.01, "maxiter": 3000, "disp": False})
                models_bomcf.append(gp)

            final_model_bomcf = ModelList(*models_bomcf)
            qEI_bomcf = qExpectedImprovement(
                    final_model_bomcf,
                    best_f=best_f,
                    sampler=sampler,
                    objective=objective
                    )
            try:
                cands = optimize_ei(qEI_bomcf, bounds, **optimize_acqf_kwargs)
            except:
                cands = None

            if cands == None:
                return
            Xnew = cands
            if Xnew.shape[0] > 0:
                here = c_batched(Xnew)
                train_X["bomcf"] = torch.cat(
                        [train_X["bomcf"], Xnew])
                train_Y["bomcf"] = torch.cat([train_Y["bomcf"], here])
                vals = objective(here)
                print("bomcf objective: ", vals)
                val = here[0][0] * here[0][2] + here[0][1] * here[0][3]
                print("bomcf regret: ", global_maxima - val)
                with open(regret_file, "a+") as f:
                    f.write(f"BOMCF -- {global_maxima - val}\n")
            else:
                with open(regret_file, "a+") as f:
                    f.write(f"BOMCF -- None\n")
            with open(runtime_file, "a+") as f:
                f.write(f"BOMCF -- {time.monotonic() - tic}\n")

        p = multiprocessing.Process(target=lambda: hogp(train_X, train_Y, best_f["ei_hogp_cf"]))
        p.start()
        p.join(120)
        if p.is_alive():
            p.kill()
            print("killed hogp after 120sec")
            p.join()
            with open(regret_file, "a+") as f:
                f.write("HOGP -- failed\n")
            with open(runtime_file, "a+") as f:
                f.write("HOGP -- 120.0\n")

        p = multiprocessing.Process(target=lambda: bomcf(train_X, train_Y, best_f["bomcf"]))
        p.start()
        p.join(40)
        if p.is_alive():
            p.kill()
            print("killed bomcf after 40sec")
            p.join()
            with open(regret_file, "a+") as f:
                f.write("BOMCF -- failed\n")
            with open(runtime_file, "a+") as f:
                f.write("BOMCF -- 40.0\n")

