import numpy as np
import time

def FW_Q(Q, c, x, verbosity, maxit, maxtime, eps, fstop, stopcr):
    gamma = 1e-4

    flagls = 0

    _, n = Q.shape

    if maxit < np.inf:
        fh = np.zeros(maxit)
        timeVec = np.zeros(maxit)
    else:
        fh = np.zeros(100 * n)
        timeVec = np.zeros(100 * n)

    it = 1

    tstart = time.time()

    while it <= maxit and flagls == 0:

        Qx = Q @ x
        xQx = x.T @ Qx
        cx = c.T @ x

        if it == 1:
            fx = 0.5 * xQx - cx
            timeVec[it - 1] = 0
        else:
            timeVec[it - 1] = time.time() - tstart

        fh[it - 1] = fx

        # gradient evaluation
        g = Qx - c

        if timeVec[it - 1] > maxtime:
            break

        # solution of FW problem
        istar = np.argmin(g)
        xstar = np.zeros(n)
        xstar[istar] = 1.0

        # direction calculation
        d = xstar - x
        gnr = g.T @ d

        # stopping criteria and test for termination
        if stopcr == 1:
            if fx <= fstop:
                break
        elif stopcr == 2:
            if gnr >= -eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Armijo search
        alpha = 1
        ref = gamma * gnr

        while True:
            fz = 0.5 * ((1 - alpha)**2 * xQx + 2 * alpha * (1 - alpha) * Qx[istar] +
                        alpha**2 * Q[istar, istar]) - ((1 - alpha) * cx + alpha * c[istar])

            if fz <= fx + alpha * ref:
                z = x + alpha * d
                break
            else:
                alpha *= 0.5

            if alpha <= 1e-20:
                z = x
                fz = fx
                flagls = 1
                it -= 1
                break

        x = z
        fx = fz

        if verbosity > 0:
            print(f"-----------------** {it} **------------------")
            print(f"f(x)     = {fx}")

        it += 1

    ttot = time.time() - tstart

    if it < len(fh):
        fh = fh[:it]
        timeVec = timeVec[:it]

    return x, it, fx, ttot, fh, timeVec
