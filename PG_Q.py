import numpy as np
import time

def PG_Q(Q, c, x, verbosity, maxit, maxtime, eps, fstop, stopcr):
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

    def proj_simplex_vector(y):
        return np.maximum(y - np.maximum((np.cumsum(np.sort(y, axis=0)[::-1], axis=0) - 1) / np.arange(1, len(y) + 1), 0), 0)

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

        # compute direction
        d = proj_simplex_vector(x - g) - x

        # stopping criteria and test for termination
        if stopcr == 1:
            if fx <= fstop:
                break
        elif stopcr == 2:
            istar = np.argmin(g)
            xstar = np.zeros(n)
            xstar[istar] = 1.0
            if g.T @ (xstar - x) >= -eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Armijo search
        alpha = 1.0
        gd = g.T @ d
        ref = gamma * gd

        dQd = d.T @ Q @ d

        while True:
            fz = fx + alpha * (gd + 0.5 * alpha * dQd)

            if fz <= fx + alpha * ref:
                z = x + alpha * d
                break
            else:
                alpha *= 0.5

            if alpha <= 1e-20:
                z = x
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
