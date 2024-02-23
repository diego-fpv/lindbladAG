import numpy as np
import os
import time
import types
import warnings
import scipy.integrate
from qutip.qobj import Qobj, isket
from qutip.states import basis
from qutip.superoperator import vec2mat, mat2vec, lindblad_dissipator, liouvillian
from qutip.expect import expect
from qutip.solver import Options, Result, config, _solver_safety_check
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.cy.openmp.utilities import check_use_openmp
import qutip.settings as qset

def evaluate(f: np.ndarray, x: float) -> np.ndarray:
    """ 
    Evaluate a symmetric array of functions at a given value.
    Input:
        - f: array of functions to be evaluated. A 1d array is interpreted as a diagonal matrix.
        - x: value at which to evaluate (only one value, not an array of values)
    Returns:
        - square symmetric array of floats.
    """

    # only diagonal
    if len(f.shape) == 1:
        # evaluate each (diagonal) matrix element
        fEval = [fi(x) for fi in f]

        return np.diag(fEval)
    
    # full array
    else:
        # consistency checks
        assert len(f.shape) == 2
        assert f.shape[0] == f.shape[1]

        # create empty variable to fill
        fOut = np.zeros(f.shape)
        for i, fi in enumerate(f):
            for j, fij in enumerate(fi[i:], start=i):
                # fill symmetric array
                fOut[i, j] = fOut[j, i] = fij(x)
        
        return fOut

def mesolveAG(H, psi0, tlist, a_ops=[], J=None, L=None, e_ops=[], c_ops=[],
              args={}, use_secular=False, sec_cutoff = 0.1, options=None,
              progress_bar=None, _safe_mode=True, verbose=False):
    """
    Solves for the dynamics of a system using the Bloch-Redfield master equation,
    given an input Hamiltonian, Hermitian bath-coupling terms and their associated
    spectrum functions, as well as possible Lindblad collapse operators.

    For time-independent systems, the Hamiltonian must be given as a Qobj,
    whereas the bath-coupling terms (a_ops), must be written as a nested list
    of operator - spectrum function pairs, where the frequency is specified by
    the `w` variable.

    *Example*

        a_ops = [[a+a.dag(),lambda w: 0.2*(w>=0)]]

    For time-dependent systems, the Hamiltonian, a_ops, and Lindblad collapse
    operators (c_ops), can be specified in the QuTiP string-based time-dependent
    format.  For the a_op spectra, the frequency variable must be `w`, and the
    string cannot contain any other variables other than the possibility of having
    a time-dependence through the time variable `t`:

    *Example*

        a_ops = [[a+a.dag(), '0.2*exp(-t)*(w>=0)']]

    It is also possible to use Cubic_Spline objects for time-dependence.  In
    the case of a_ops, Cubic_Splines must be passed as a tuple:

    *Example*

        a_ops = [ [a+a.dag(), ( f(w), g(t)] ]

    where f(w) and g(t) are strings or Cubic_spline objects for the bath
    spectrum and time-dependence, respectively.

    Finally, if one has bath-couplimg terms of the form
    H = f(t)*a + conj[f(t)]*a.dag(), then the correct input format is

    *Example*

              a_ops = [ [(a,a.dag()), (f(w), g1(t), g2(t))],... ]

    where f(w) is the spectrum of the operators while g1(t) and g2(t)
    are the time-dependence of the operators `a` and `a.dag()`, respectively

    Parameters
    ----------
    H : Qobj / list
        System Hamiltonian given as a Qobj or
        nested list in string-based format.

    psi0: Qobj
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution

    a_ops : list
        Nested list of Hermitian system operators that couple to
        the bath degrees of freedom, along with their associated
        spectra.

    e_ops : list
        List of operators for which to evaluate expectation values.

    c_ops : list
        List of system collapse operators, or nested list in
        string-based format.

    args : dict
        Placeholder for future implementation, kept for API consistency.

    use_secular : bool {True}
        Use secular approximation when evaluating bath-coupling terms.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation.

    tol : float {qutip.setttings.atol}
        Tolerance used for removing small values after
        basis transformation.

    spectra_cb : list
        DEPRECIATED. Do not use.

    options : :class:`qutip.solver.Options`
        Options for the solver.

    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.
    """

    #This allows for passing a list of time-independent Qobj
    #as allowed by mesolve
    if isinstance(H, list):
        if np.all([isinstance(h,Qobj) for h in H]):
            H = sum(H)

    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops = [e for e in e_ops.values()]

    if _safe_mode:
        _solver_safety_check(H, psi0, a_ops+c_ops, e_ops, args)

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    if options is None:
        options = Options()

    if (not options.rhs_reuse) or (not config.tdfunc):
        # reset config collapse and time-dependence flags to default values
        config.reset()

    #check if should use OPENMP
    check_use_openmp(options)

    T, ekets = lindbladianAG(H=H, a_ops=a_ops, J=J, L=L, c_ops=c_ops,
                             use_secular=use_secular, sec_cutoff=sec_cutoff)

    output = Result()
    output.times = tlist

    results = mesolveAG_solve(T, ekets, psi0, tlist, e_ops, options,
                              progress_bar=progress_bar)

    if e_ops:
        output.expect = results
    else:
        output.states = results

    return output
    
def lindbladianAG(H, a_ops, J, L=None, c_ops=[], use_secular=False, sec_cutoff=0.1):
    """
    Description
    Inputs:
    Returns:
    Comments:
        With the formulation of {ref}, the relevant space of indices is that of transitions (i and j in {ref}). However, not all potentially possible transitions (N^2 of them) really have non-zero matrix elements in general, and it is a good strategy to filter them such that only transitions with non-vanishing matrix element are kept.
    """

    # sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")
    
    for a in a_ops:
        if not isinstance(a, Qobj) or not a.isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")
    
    # eigenbasis:
    evals, ekets = H.eigenstates()
    # diagonalized Hamiltonian:
    Heb = H.transform(ekets)
    N = len(evals)
    K = len(a_ops)

    # transitions:    
    #   transition operators as arrays
    a_arr = np.array([a_ops[k].transform(ekets).full() for k in range(K)])
    #   total matrix element for each transition (to filter the transitions that need to be summed over)
    a_tot = np.linalg.norm(a_arr, axis=0)
    #   the transition |ni><mi| has an associated transition operator <ni|A|mi> = Ai and energy wi = Emi - Eni
    matElements = np.asarray([a_arr[:, n, m] for n in range(N) for m in range(N) if a_tot[n, m] != 0.0])
    transitions = [(n, m) for n in range(N) for m in range(N) if a_tot[n, m] != 0.0]
    W = np.array([(H[m, m] - H[n, n]).real for n, m in transitions])
    T = len(W)

    # cutoff for secular approximation, if needed
    dw_min = np.abs(W[W.nonzero()]).min()
    
    # Liouvillian tensor. System's Hamiltonian + dissipation from c_ops (if given)
    Liouvillian = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])

    # Gamma(w) evaluated at the transition frequencies (see {ref})
    Gammaw = np.zeros((K, K, T))
    for i in range(T):
        Gammaw[:, :, i] = 2 * np.pi * evaluate(J, W[i])

    # Kossakowski matrix only: 
    # (dispatched like this to avoid looping over transitions 2 times)
    if L == None or L == []:
        Kmatrix = np.zeros((T, T), dtype=complex)
        for i, Ai in enumerate(matElements):
            # only check use_secular once per transition i
            if use_secular:
                transitionSubsetSec = np.where(np.abs(W[i] - W) < dw_min * sec_cutoff)[0]
                matElementsSec = matElements[transitionSubsetSec]
            else:
                transitionSubsetSec = range(T)
                matElementsSec = matElements
            for j, Aj in zip(transitionSubsetSec, matElementsSec):
                Gammai = np.einsum("a,b,ab->", Ai.conj(), Aj, Gammaw[:, :, i])
                Gammaj = np.einsum("a,b,ab->", Ai.conj(), Aj, Gammaw[:, :, j])
                # geometric mean for the Kossakowski matrix
                Kmatrix[i, j] = np.sqrt(Gammai) * np.sqrt(Gammaj)
    
    # Both Kossakowski matrix and Lamb shift matrix
    else:
        Kmatrix = np.zeros((T, T), dtype=complex)
        Lmatrix = np.zeros((T, T), dtype=complex) 

        # evaluate spectral function L:
        Lambdaw = np.zeros((K, K, T))
        for i in range(T):
            Lambdaw[:, :, i] = evaluate(L, W[i])

        for i, Ai in enumerate(matElements):
            # only check use_secular once per transition i
            if use_secular:
                transitionSubsetSec = np.where(np.abs(W[i] - W) < dw_min * sec_cutoff)[0]
                matElementsSec = matElements[transitionSubsetSec]
            else:
                transitionSubsetSec = range(T)
                matElementsSec = matElements

            for j, Aj in zip(transitionSubsetSec, matElementsSec):
                # geometric mean for the Kossakowski matrix
                Gammai = np.einsum("a,b,ab->", Ai.conj(), Aj, Gammaw[:, :, i])
                Gammaj = np.einsum("a,b,ab->", Ai.conj(), Aj, Gammaw[:, :, j])
                Kmatrix[i, j] = np.sqrt(Gammai) * np.sqrt(Gammaj)
                # arithmetic mean for the Lamb shift matrix
                Lambdai = np.einsum("a,b,ab->", Ai.conj(), Aj, Lambdaw[:, :, i])
                Lambdaj = np.einsum("a,b,ab->", Ai.conj(), Aj, Lambdaw[:, :, j])
                Lmatrix[i, j] = 0.5 * (Lambdai + Lambdaj)

    # diagonalize Kossakowski matrix and keep only positive rates
    rates, vecs = np.linalg.eigh(Kmatrix)
    idxPositive = rates > 0
    rates, vecs = rates[idxPositive], vecs[:, idxPositive]
    
    # fill Lindblad superoperator:
    for rate, vec in zip(rates, vecs.T):
        sigmaj = 0
        for (nj, mj), vj in zip(transitions, vec):
            sigmaj += vj.conj() * basis(N, nj) * basis(N, mj).dag()
        Liouvillian += rate * lindblad_dissipator(sigmaj)

    # calculate Lamb shift Hamiltonian and commutator superoperator
    if L != None and L != []:
        pass


    return Liouvillian, ekets

def mesolveAG_solve(T, ekets, rho0, tlist, e_ops=[], options=None, progress_bar=None):
    """
    Evolve the ODEs defined by Bloch-Redfield master equation. The
    Bloch-Redfield tensor can be calculated by the function
    :func:`bloch_redfield_tensor`.

    Parameters
    ----------

    T : :class:`qutip.qobj`
        Lindbladian superoperator of the arithmetic-geometric master equation (see {ref}).

    ekets : array of :class:`qutip.qobj`
        Array of kets that make up a basis tranformation for the eigenbasis.

    rho0 : :class:`qutip.qobj`
        Initial density matrix.

    tlist : *list* / *array*
        List of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    options : :class:`qutip.solver.Options`
        Options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains either
        an *array* of expectation values for the times specified by `tlist`.

    """

    if options is None:
        options = Options()

    if options.tidy:
        T.tidyup()

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()

    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]
    result_list = []

    #
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis
    #
    rho_eb = rho0.transform(ekets)
    e_eb_ops = [e.transform(ekets) for e in e_ops]

    for e_eb in e_eb_ops:
        if e_eb.isherm:
            result_list.append(np.zeros(n_tsteps, dtype=complex))
        else:
            result_list.append(np.zeros(n_tsteps, dtype=complex))

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho_eb.full())
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(T.data.data, T.data.indices, T.data.indptr)
    r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step, max_step=options.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    dt = np.diff(tlist)
    progress_bar.start(n_tsteps)
    for t_idx, _ in enumerate(tlist):
        progress_bar.update(t_idx)
        if not r.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")

        rho_eb.data = dense2D_to_fastcsr_fmode(vec2mat(r.y), rho0.shape[0], rho0.shape[1])

        # calculate all the expectation values, or output rho_eb if no
        # expectation value operators are given
        if e_ops:
            rho_eb_tmp = Qobj(rho_eb)
            for m, e in enumerate(e_eb_ops):
                result_list[m][t_idx] = expect(e, rho_eb_tmp)
        else:
            result_list.append(rho_eb.transform(ekets, True))

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])
    progress_bar.finished()
    return result_list