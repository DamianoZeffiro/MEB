import numpy as np
import time
from matplotlib import pyplot as plt
from FW_Q import FW_Q
from FWAW_Q import FWAW_Q
from PG_Q import PG_Q

def Main_MEB():
    # Number of samples
    m = 2 ** 5

    # Number of variables
    n = 2 ** 12

    # Number of runs for each instance:
    nrun = 1

    # Number of solvers:
    nsolvers = 3

    maxit = 3000
    maxtime = 100

    frun_cell = [[None for _ in range(nsolvers)] for _ in range(nrun)]
    timeVectot_cell = [[None for _ in range(nsolvers)] for _ in range(nrun)]

    fstop = np.zeros(nrun)

    for krun in range(nrun):
        # Generation of the instance:

        # Seed changes at every run to generate starting point
        np.random.seed(krun)

        x0 = np.zeros(n)
        x0[0] = 1e0

        Q = np.random.randn(m, n)
        c = np.sum(Q ** 2, axis=0)
        Q = 2e0 * (Q.T @ Q)

        count = 0
        eps = 1e-6
        stopcr = 2

        print("****************")
        print("* AWAY STEP FW *")
        print("****************")

        xfwaw, iterfwaw, fxfwaw, tottimefwaw, frun_cell[krun][count], timeVectot_cell[krun][count] = FWAW_Q(Q, c, x0, 0,
                                                                                                            maxit,
                                                                                                            maxtime,
                                                                                                            eps,
                                                                                                            fstop[krun],
                                                                                                            stopcr)

        # Print results:
        print(f"0.5*xQX - cx = {fxfwaw:.3e}")
        print(f"Number of non-zero components of x = {np.sum(np.abs(xfwaw) >= 0.0001)}")
        print(f"Number of iterations = {iterfwaw}")
        print(f"CPU time = {tottimefwaw:.3e}")

        count += 1

        print("*****************")
        print("*  FW STANDARD  *")
        print("*****************")

        xfw, iterfw, fxfw, tottimefw, frun_cell[krun][count], timeVectot_cell[krun][count] = FW_Q(Q, c, x0, 0, maxit,
                                                                                                  maxtime, eps,
                                                                                                  fstop[krun], stopcr)

        # Print results:
        print(f"0.5*xQX - cx = {fxfw:.3e}")
        print(f"Number of non-zero components of x = {np.sum(np.abs(xfw) >= 0.0001)}")
        print(f"Number of iterations = {iterfw}")
        print(f"CPU time = {tottimefw:.3e}")

        count += 1

        print("*****************")
        print("*      PG       *")
        print("*****************")

        x_pg, iter_pg, fx_pg, tottime_pg, frun_cell[krun][count], timeVectot_cell[krun][count] = PG_Q(Q, c, x0, 0,
                                                                                                      maxit, maxtime,
                                                                                                      eps, fstop[krun],
                                                                                                      stopcr)

        # Print results:
        print(f"0.5*xQX - cx = {fx_pg:.3e}")
        print(f"Number of non-zero components of x = {np.sum(np.abs(x_pg) >= 0.0001)}")
        print(f"Number of iterations = {iter_pg}")
        print(f"CPU time = {tottime_pg:.3e}")

        fstop[krun] = min([fxfwaw, fxfw, fx_pg])

    maxit = 0
    for i in range(nsolvers):
        for krun in range(nrun):
            maxit = max(maxit, len(frun_cell[krun][i]))

    frun = np.zeros((nrun, nsolvers, maxit))
    timerun = np.zeros((nrun, nsolvers, maxit))
    for i in range(nsolvers):
        for krun in range(nrun):
            niter = len(frun_cell[krun][i])
            fx = frun_cell[krun][i][-1]
            t = timeVectot_cell[krun][i][-1]
            frun[krun, i, :niter] = frun_cell[krun][i]
            frun[krun, i, niter:] = fx
            timerun[krun, i, :niter] = timeVectot_cell[krun][i]
            timerun[krun, i, niter:] = t

    del frun_cell, timeVectot_cell, Q, c

    frun_plot = np.zeros((nsolvers, maxit))
    timerun_plot = np.zeros((nsolvers, maxit))
    for i in range(nsolvers):
        for krun in range(nrun):
            frun_plot[i, :] += np.maximum(0, frun[krun, i, :] - fstop[krun])
            timerun_plot[i, :] += timerun[krun, i, :]
    frun_plot /= nrun
    timerun_plot /= nrun

    plot_threshold = 1e-20

    # AFW
    plot_fwaw = np.maximum(frun_plot[0, :maxit], plot_threshold)
    plot_tfwaw = timerun_plot[0, :maxit]

    # FW
    plotfw = np.maximum(frun_plot[1, :maxit], plot_threshold)
    plottfw = timerun_plot[1, :maxit]

    # PG
    plot_pg = np.maximum(frun_plot[2, :maxit], plot_threshold)
    plot_tpg = timerun_plot[2, :maxit]

    # All Solvers
    plt.figure()
    plt.semilogy(plot_tfwaw, plot_fwaw, 'b-', label='AFW')
    plt.semilogy(plottfw, plotfw, 'r-', label='FW')
    plt.semilogy(plot_tpg, plot_pg, 'g-', label='PG')
    plt.title('FW variants & PG - objective function error')
    plt.legend(['AFW', 'FW', 'PG'])
    plt.xlim([0, 5])
    plt.ylim([1e-10, 1e5])
    plt.savefig(f'Plot_m_{m}_n_{n}_all_solvers.png')
    plt.show()

Main_MEB()
