import numpy as np
import scipy.integrate
from qutip.qobj import Qobj, isket
from qutip.states import basis
from qutip.superoperator import vec2mat, vec2mat_index, mat2vec, lindblad_dissipator, liouvillian, spre, spost
from qutip.expect import expect
from qutip.solver import Options, Result, config, _solver_safety_check
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode, arr_coo2fast
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.cy.openmp.utilities import check_use_openmp
import qutip.settings as qset


# -------------------------------------------------------------------------------
# Evaluate an array of callable objects at a given point.
# 
def evaluate(f: np.ndarray, x: (float | int)) -> np.ndarray:
    """ 
    Evaluate a symmetric array of functions at a given value.

    Parameters
    ----------

    f : np.ndarray
        Array of functions to be evaluated. 
        A 1d array is interpreted as a diagonal matrix.
    
    x : float / int
        value at which to evaluate (only one value, 
        not an array of values)
    
    Returns
    -------
        -
    fOut : np.ndarray 
        Square symmetric array of floats.
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


# -------------------------------------------------------------------------------
# Solve the arithmetic-geometric master equation, with a similar signature to
# brmesolve (prior to QuTiP 4.3). Only time-independent problems are currently
# supported. 
# 
def mesolveAG(H, psi0, tlist, a_ops=[], J=None, L=None, e_ops=[], c_ops=[],
              use_secular=False, sec_cutoff = 0.1, options=None, tol=qset.atol,
              progress_bar=None, _safe_mode=True, verbose=False):
    """
    Solves for the dynamics of a system using the arithmetic-geometric master equation,
    given an input Hamiltonian, Hermitian bath-coupling terms and their associated
    spectral functions, as well as other possible Lindblad collapse operators.

    The Hamiltonian must be given as a Qobj, the bath-coupling terms (a_ops)
    must be written as a list of operators, and the spectral functions have to be passed
    as 1D or 2D square arrays of callable objects.

    *Example*

        a_ops = [a1+a1.dag(), a2+a2.dag()]
        J = np.array([lambda x, ci=i: i*(x>0) for i in [1, 2]])
        L = np.array([lambda x, ci=i: 3*i*(x>0) for i in [1, 2]])

    The implementation leaves J and L as independent parameters, but in reality these are Hilbert conjugates of each other. This should be taken into account while creating the arrays before calling this function.
        
    Only time-independent Hamiltonians are currently supported.

    Parameters
    ----------
    H : Qobj / list
        System Hamiltonian given as a Qobj or
        nested list in string-based format.

    psi0: Qobj
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution.

    a_ops : list
        List of Hermitian system operators that couple to
        the bath degrees of freedom.

    J : np.ndarray
        Array of functions that describe the coupling between
        the system and the environment (dissipative part).

    J : np.ndarray
        Array of functions that describe the coupling between
        the system and the environment (energy shift part).

    e_ops : list
        List of operators for which to evaluate expectation values.

    c_ops : list
        List of system collapse operators, or nested list in
        string-based format.

    use_secular : bool {True}
        Use secular approximation when evaluating bath-coupling terms.

    sec_cutoff : float {0.1}
        Cutoff for secular approximation.

    tol : float {qutip.setttings.atol}
        Tolerance used for removing small values after
        basis transformation.

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
        _solver_safety_check(H, psi0, a_ops+c_ops, e_ops,)

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


# -------------------------------------------------------------------------------
# Function for calculating the Lindbladian of the arithmetic-geometric master 
# equation.
# 
def lindbladianAG(H, a_ops, J, L=None, c_ops=[], use_secular=False, sec_cutoff=0.1):
    """
    Compute the Lindbladian for a system given its Hamiltonian, a list of system operators coupling it to the baths and the corresponding spectral functions.

    .. note::

        This tensor generation requires a time-independent Hamiltonian.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        System Hamiltonian.

    a_ops : list of :class:`qutip.qobj`
        List of system operators that couple to the environment.

    J : array of callback functions
        Array of callback functions that evaluate the spectral density
        at a given frequency.
    
    L : array of callback functions
        Array of callback functions that evaluate the integral of the
        spectral density at a given frequency.

    c_ops : list of :class:`qutip.qobj`
        List of system collapse operators.

    use_secular : bool
        Flag (True of False) that indicates if the secular approximation should
        be used.
    
    sec_cutoff : float
        Parameter to control the secular approximation cutoff frequency.

    Returns
    -------

    Liouvillian, ekets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        Liouvillian is the Liouvillian tensor and ekets is a list eigenstates of the
        Hamiltonian.

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

    # Liouvillian tensor. System's Hamiltonian + dissipation from c_ops (if given)
    Liouvillian = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])

    # transitions: With the formulation of {ref}, the relevant space of indices is that 
    # of transitions (i and j in {ref}). However, not all potentially possible transitions 
    # (N^2 of them) really have non-zero matrix elements in general, and it is a good 
    # strategy to filter them such that only transitions with non-vanishing matrix 
    # element are kept.
    #   transition operators as arrays
    a_arr = np.array([a_ops[k].transform(ekets).full() for k in range(K)])
    #   total matrix element for each transition (to filter the transitions that need to be summed over)
    a_tot = np.linalg.norm(a_arr, axis=0)
    #   the transition |ni><mi| has an associated transition operator <ni|A|mi> = Ai 
    #   and energy wi = Emi - Eni
    matElements = np.asarray([a_arr[:, n, m] for n in range(N) for m in range(N) if a_tot[n, m] != 0.0])
    transitions = [(n, m) for n in range(N) for m in range(N) if a_tot[n, m] != 0.0]
    W = np.array([(Heb[m, m] - Heb[n, n]).real for n, m in transitions])
    T = len(W)

    # cutoff for secular approximation, in case it is needed
    dw_min = np.abs(W[W.nonzero()]).min() * sec_cutoff

    # Gamma(w) evaluated at the transition frequencies (see {ref})
    Gammaw = np.zeros((K, K, T))
    for i in range(T):
        Gammaw[:, :, i] = 2 * np.pi * evaluate(J, W[i])

    # Kossakowski matrix only: 
    # (dispatched like this to avoid looping over transitions 2 times)
    if not isinstance(L, np.ndarray):
        Kmatrix = np.zeros((T, T), dtype=complex)
        for i, Ai in enumerate(matElements):
            # only check use_secular once per transition i
            if use_secular:
                transitionSubsetSec = np.where(np.abs(W[i] - W) < dw_min)[0]
                matElementsSec = matElements[transitionSubsetSec]
            else:
                transitionSubsetSec = range(T)
                matElementsSec = matElements
            # loop over the secularized (or not) subset of j transitions
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
            # loop over the secularized (or not) subset of j transitions
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
    Hls = np.zeros((N, N), dtype=complex)
    if isinstance(L, np.ndarray):
        for i, (ni, mi) in enumerate(transitions):
            for j, (nj, mj) in enumerate(transitions):
                # transitions i and j connect states mi and mj: |mi><ni|nj><mj|
                if ni == nj:
                    Hls[mi, mj] += Lmatrix[i, j]
        Hls = Qobj(Hls)
        Liouvillian += -1j * (spre(Hls) - spost(Hls))

    return Liouvillian, ekets


# -------------------------------------------------------------------------------
# Evolution of the arithmetic-geometric master equation given the lindbladian
# tensor.
# 
def mesolveAG_solve(T, ekets, rho0, tlist, e_ops=[], options=None, progress_bar=None):

    """
    Evolve the ODEs defined by the arithmetic-geometric master equation. The
    Liouvillian tensor can be calculated by the function
    :func:`liouvillianAG`.

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


# -------------------------------------------------------------------------------
# Solve the Bloch-Redfield equation including the Lamb shift with a minimal 
# change from QuTiP's old version (< 4.3)
# 
def old_brmesolve(H, psi0, tlist, a_ops, e_ops=[], spectra_cb=[], c_ops=[],
              args={}, options=Options(), use_secular=False, sec_cutoff=0.1,
              _safe_mode=True):
    """
    Solve the dynamics for a system using the Bloch-Redfield master equation.

    .. note::

        This solver does not currently support time-dependent Hamiltonians.

    Parameters
    ----------

    H : :class:`qutip.Qobj`
        System Hamiltonian.

    rho0 / psi0: :class:`qutip.Qobj`
        Initial density matrix or state vector (ket).

    tlist : *list* / *array*
        List of times for :math:`t`.

    a_ops : list of :class:`qutip.qobj`
        List of system operators that couple to bath degrees of freedom.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    c_ops : list of :class:`qutip.qobj`
        List of system collapse operators.

    args : *dictionary*
        Placeholder for future implementation, kept for API consistency.

    options : :class:`qutip.solver.Options`
        Options for the solver.

    Returns
    -------

    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.
    """

    if _safe_mode:
        _solver_safety_check(H, psi0, a_ops, e_ops, args)

    R, ekets = old_bloch_redfield_tensor(H, a_ops, spectra_cb, c_ops)

    output = Result()
    output.solver = None
    output.times = tlist

    results = old_bloch_redfield_solve(R, ekets, psi0, tlist, e_ops, options)

    if e_ops:
        output.expect = results
    else:
        output.states = results

    return output


# -----------------------------------------------------------------------------
# Evolution of the Bloch-Redfield master equation given the Bloch-Redfield
# tensor.
#
def old_bloch_redfield_solve(R, ekets, rho0, tlist, e_ops=[], options=None):
    """
    Evolve the ODEs defined by Bloch-Redfield master equation. The
    Bloch-Redfield tensor can be calculated by the function
    :func:`bloch_redfield_tensor`.

    Parameters
    ----------

    R : :class:`qutip.qobj`
        Bloch-Redfield tensor.

    ekets : array of :class:`qutip.qobj`
        Array of kets that make up a basis tranformation for the eigenbasis.

    rho0 : :class:`qutip.qobj`
        Initial density matrix.

    tlist : *list* / *array*
        List of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    options : :class:`qutip.Qdeoptions`
        Options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`.

    """

    if options is None:
        options = Options()

    if options.tidy:
        R.tidyup()

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
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step, max_step=options.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    dt = np.diff(tlist)
    for t_idx, _ in enumerate(tlist):

        if not r.successful():
            break

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

    return result_list


# -----------------------------------------------------------------------------
# Functions for calculating the Bloch-Redfield tensor for a time-independent
# system.
#
def old_bloch_redfield_tensor(H, a_ops, spectra_cb=[], c_ops=[],
                    use_secular=False, sec_cutoff=0.1):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment.

    .. note::

        This tensor generation requires a time-independent Hamiltonian.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        System Hamiltonian.

    a_ops : list of :class:`qutip.qobj`
        List of system operators that couple to the environment.

    spectra_cb : list of callback functions
        List of callback functions that evaluate the noise power spectrum
        at a given frequency.

    c_ops : list of :class:`qutip.qobj`
        List of system collapse operators.

    use_secular : bool
        Flag (True of False) that indicates if the secular approximation should
        be used.

    Returns
    -------

    R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        R is the Bloch-Redfield tensor and kets is a list eigenstates of the
        Hamiltonian.

    """
    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")
        
    for a in a_ops:
        if not isinstance(a, Qobj) or not a.isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    if c_ops is None:
        c_ops = []

    # use the eigenbasis
    evals, ekets = H.eigenstates()

    N = len(evals)
    K = len(a_ops)
    
    # only Lindblad collapse terms
    if K==0:
        Heb = H.transform(ekets)
        L = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])
        return L, ekets
    
    
    A = np.array([a_ops[k].transform(ekets).full() for k in range(K)])
    Sw = np.zeros((K, N, N), dtype=complex)

    # pre-calculate matrix elements and spectral densities
    W = np.real(evals[:,np.newaxis] - evals[np.newaxis,:])

    for k in range(K):
        # do explicit loops here in case spectra_cb[k] can not deal with array arguments
        for n in range(N):
            for m in range(N):
                Sw[k, n, m] = spectra_cb[k](W[n, m])

    dw_min = np.abs(W[W.nonzero()]).min()

    # pre-calculate mapping between global index I and system indices a,b
    Iabs = np.empty((N*N,3),dtype=int)
    for I, Iab in enumerate(Iabs):
        # important: use [:] to change array values, instead of creating new variable Iab
        Iab[0]  = I
        Iab[1:] = vec2mat_index(N, I)

    # unitary part + dissipation from c_ops (if given):
    Heb = H.transform(ekets)
    L = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])
    
    # dissipative part:
    rows = []
    cols = []
    data = []
    for I, a, b in Iabs:
        # only check use_secular once per I
        if use_secular:
            # only loop over those indices J which actually contribute
            Jcds = Iabs[np.where(np.abs(W[a, b] - W[Iabs[:,1], Iabs[:,2]]) < dw_min * sec_cutoff)]
        else:
            Jcds = Iabs
        for J, c, d in Jcds:
            elem = 0+0j
            # summed over k, i.e., each operator coupling the system to the environment
            # notice the last .conj() to properly account for the Lamb shift
            elem += 0.5 * np.sum(A[:, a, c] * A[:, d, b] * (Sw[:, c, a] + Sw[:, d, b].conj()))
            if b==d:
                #                  sum_{k,n} A[k, a, n] * A[k, n, c] * Jw[k, c, n])
                elem -= 0.5 * np.sum(A[:, a, :] * A[:, :, c] * Sw[:, c, :])
            if a==c:
                #                  sum_{k,n} A[k, d, n] * A[k, n, b] * Jw[k, d, n])
                #                  notice the last .conj() to properly account for the 
                #                  Lamb shift
                elem -= 0.5 * np.sum(A[:, d, :] * A[:, :, b] * Sw[:, d, :].conj())
            if elem != 0 + 0j:
                rows.append(I)
                cols.append(J)
                data.append(elem)

    R = arr_coo2fast(np.array(data, dtype=complex),
                    np.array(rows, dtype=np.int32),
                    np.array(cols, dtype=np.int32), N**2, N**2)
    
    L.data = L.data + R
    
    return L, ekets

