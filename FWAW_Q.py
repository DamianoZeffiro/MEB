import numpy as np
import time

def FWAW_Q(Q, c, x, verbosity, maxit, maxtime, eps, fstop, stopcr):
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

        # solution of Away step problem
        indcc = np.where(x > 0)[0]
        istaraw = np.argmax(g[indcc])

        xstar2 = np.zeros(n)
        indaw = indcc[istaraw]
        xstar2[indaw] = 1.0

        # directions calculation
        dFW = xstar - x
        dAS = x - xstar2

        p1 = g.T @ dFW
        p2 = g.T @ dAS

        # choice of the search direction
        if p1 <= p2:
            d = dFW
            alpham = 1.0
            gnr = p1
            cdir = 1
        else:
            d = dAS
            alpham = x[indaw] / (1 - x[indaw])
            gnr = p2
            cdir = 2

        # stopping criteria and test for termination
        if stopcr == 1:
            if fx <= fstop:
                break
        elif stopcr == 2:
            if p1 >= -eps:
                break
        else:
            raise ValueError("Unknown stopping criterion")

        # Armijo search
        alpha = alpham
        ref = gamma * gnr

        while True:

            # Smart computation of the o.f. at the trial point
            if cdir == 1:
                fz = 0.5 * ((1 - alpha)**2 * xQx + 2 * alpha * (1 - alpha) * Qx[istar] +
                            alpha**2 * Q[istar, istar]) - ((1 - alpha) * cx + alpha * c[istar])
            else:
                fz = 0.5 * ((1 + alpha)**2 * xQx - 2 * alpha * (1 + alpha) * Qx[indaw] +
                            alpha**2 * Q[indaw, indaw]) - ((1 + alpha) * cx - alpha * c[indaw])

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
            print("-----------------** {} **------------------".format(it))
            print("f(x)     = {}".format(fx))

        it += 1

    ttot = time.time() - tstart

    if it < len(fh):
        fh = fh[:it]
        timeVec = timeVec[:it]

    return x, it, fx, ttot, fh, timeVec
