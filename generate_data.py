import argparse
import os
import random
import time

import numpy as np
import pyamg
from joblib import Parallel, delayed
from retry import retry
from scipy import interpolate
from tqdm import tqdm


def boundary_generate(size, min_b, max_b, x_range=[0, 1], y_range=[0, 1], split=4):
    x = np.linspace(x_range[0], x_range[1], split)
    y = np.linspace(y_range[0], y_range[1], split)

    x_new = np.linspace(x_range[0], x_range[1], size)
    y_new = np.linspace(y_range[0], y_range[1], size)

    z = [np.random.uniform(min_b, max_b, (split)) for _ in range(split)]
    f_scipy = []

    z[3][0] = z[0][0]
    z[3][split - 1] = z[1][0]
    z[1][split - 1] = z[2][split - 1]
    z[2][0] = z[0][split - 1]

    for i in range(split):
        if i == 0 or i == 2:
            f_scipy.append(interpolate.interp1d(x, z[i], kind="cubic"))

        else:
            f_scipy.append(interpolate.interp1d(y, z[i], kind="cubic"))

    boundary = [[], [], [], []]

    boundary[0], boundary[2] = f_scipy[0](x_new), f_scipy[2](x_new)
    boundary[1], boundary[3] = f_scipy[1](y_new), f_scipy[3](y_new)
    return boundary


def RHS_coarse_grid(N, min_v=-1, max_v=1, r=1, x_range=[0, 1], y_range=[0, 1]):
    x = np.linspace(x_range[0], x_range[1], N)
    y = np.linspace(y_range[0], y_range[1], N)
    z = np.random.uniform(min_v * r, max_v * r, (N, N))
    return x, y, z


def RHS_upsampling(size, x, y, z, x_range=[0, 1], y_range=[0, 1]):
    x_new = np.linspace(x_range[0], x_range[1], size)
    y_new = np.linspace(y_range[0], y_range[1], size)
    h = x_new[1] - x_new[0]
    r = interpolate.RectBivariateSpline(x, y, z.T)
    f = r(x_new, y_new).T.reshape(size * size)

    return x_new, y_new, f, h


def RHS_boundary(boundary, f, fk, x, y, N, k=100):
    X, Y = np.meshgrid(x, y)
    Xu, Yu = X.ravel(), Y.ravel()

    ind_L = np.squeeze(np.where(Xu == x[1]))
    ind_R = np.squeeze(np.where(Xu == x[N - 2]))
    ind_B = np.squeeze(np.where(Yu == y[1]))
    ind_T = np.squeeze(np.where(Yu == y[N - 2]))

    f[ind_L] += boundary[0]
    f[ind_R] += boundary[1]
    f[ind_T] += boundary[2]
    f[ind_B] += boundary[3]

    fk[ind_L] += k * boundary[0]
    fk[ind_R] += k * boundary[1]
    fk[ind_T] += k * boundary[2]
    fk[ind_B] += k * boundary[3]

    return f.reshape(N, N)[1 : N - 1, 1 : N - 1].reshape((N - 2) * (N - 2)), fk


def poisson_solution(f, N):
    A = pyamg.gallery.poisson((N - 2, N - 2), format="csr")
    ml = pyamg.ruge_stuben_solver(A)
    x = ml.solve(f, tol=1e-8)

    if np.linalg.norm(f - A * x) > 1e-8:
        print("Not ok, >1e-8")
        return None

    return x


def make_sol_bound(N, sol, bound, x, y):
    sol_bound = np.zeros((N, N))
    sol_bound[1 : N - 1, 1 : N - 1] = sol.reshape(N - 2, N - 2)
    sol_bound = sol_bound.reshape(N * N)

    X, Y = np.meshgrid(x, y)
    Xu, Yu = X.ravel(), Y.ravel()
    ind_L = np.squeeze(np.where(Xu == x[0]))
    ind_R = np.squeeze(np.where(Xu == x[N - 1]))
    ind_B = np.squeeze(np.where(Yu == y[0]))
    ind_T = np.squeeze(np.where(Yu == y[N - 1]))

    sol_bound[ind_L] = bound[0]
    sol_bound[ind_R] = bound[1]
    sol_bound[ind_T] = bound[2]
    sol_bound[ind_B] = bound[3]

    return sol_bound


@retry(ValueError)
def one_cycle(shape, num, generation_directory):
    try:
        bound = boundary_generate(shape, -0.02, 0.02)

        coarse_N = random.randint(7, 12)
        x, y, z = RHS_coarse_grid(coarse_N)
        x, y, f, h = RHS_upsampling(shape, x, y, z)

        f_cut, fk = RHS_boundary(bound, h**2 * f, np.copy(f), x, y, shape)
        sol_cut = poisson_solution(f_cut, shape)
        if sol_cut is None:
            raise ValueError
    except ValueError as _:
        print("not ok, exception")
        raise

    sol = make_sol_bound(shape, sol_cut, bound, x, y)
    f = f.reshape(shape, shape)
    fk = fk.reshape(shape, shape)

    sol = sol.reshape(shape, shape)

    np.savez_compressed(f"{generation_directory}/bound_{num}", b=fk, x=sol, bound=bound)


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num", default=5000, type=int, help="num of samples")
    ap.add_argument("-s", "--shape", default=128, type=int, help="shape of data")
    ap.add_argument("-j", "--jobs", default=10, type=int, help="n_jobs (don't use >15)")

    args = vars(ap.parse_args())
    return (args["shape"], args["num"], args["jobs"])


if __name__ == "__main__":
    full_time = time.time()
    np.random.seed(24)
    shape, samples_num, n_jobs = get_args()
    print(
        f"Generation with shape = {shape}x{shape}, num = {samples_num}, {n_jobs} workers"
    )

    generation_directory = f"./data_s{shape}_n{samples_num}/"

    if not os.path.exists(generation_directory):
        os.makedirs(generation_directory)

    Parallel(n_jobs=n_jobs)(
        delayed(one_cycle)(shape, num, generation_directory)
        for num in tqdm(range(samples_num))
    )
    print("full time: ", time.time() - full_time)
