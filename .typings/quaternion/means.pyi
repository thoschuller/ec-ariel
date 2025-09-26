def mean_rotor_in_chordal_metric(R, t=None):
    """Return rotor that is closest to all R in the least-squares sense

    This can be done (quasi-)analytically because of the simplicity of
    the chordal metric function.  It is assumed that the input R values
    all are normalized (or at least have the same norm).

    Note that the `t` argument is optional.  If it is present, the
    times are used to weight the corresponding integral.  If it is not
    present, a simple sum is used instead (which may be slightly
    faster).  However, because a spline is used to do this integral,
    the number of input points must be at least 4 (one more than the
    degree of the spline).

    """
def optimal_alignment_in_chordal_metric(Ra, Rb, t=None):
    """Return Rd such that Rd*Rb is as close to Ra as possible

    This function simply encapsulates the mean rotor of Ra/Rb.

    As in the `mean_rotor_in_chordal_metric` function, the `t` argument is
    optional.  If it is present, the times are used to weight the corresponding
    integral.  If it is not present, a simple sum is used instead (which may be
    slightly faster).

    Notes
    =====
    The idea here is to find Rd such that

        ∫ |Rd*Rb - Ra|^2 dt

    is minimized.  [Note that the integrand is the distance in the chordal metric.]
    We can ensure that this quantity is minimized by multiplying Rd by an
    exponential, differentiating with respect to the argument of the exponential,
    and setting that argument to 0.  This derivative should be 0 at the minimum.
    We have

        ∂ᵢ ∫ |exp[vᵢ]*Rd*Rb-Ra|^2 dt  →  2 ⟨ eᵢ * Rd * ∫ Rb*R̄a dt ⟩₀

    where → denotes taking vᵢ→0, the symbol ⟨⟩₀ denotes taking the scalar part, and
    eᵢ is the unit quaternionic vector in the `i` direction.  The only way for this
    quantity to be zero for each choice of `i` is if

        Rd * ∫ Rb*R̄a dt

    is itself a pure scalar.  This, in turn, can only happen if either (1) the
    integral is 0 or (2) if Rd is proportional to the conjugate of the integral:

        Rd ∝ ∫ Ra*R̄b dt

    Now, since we want Rd to be a rotor, we simply define it to be the normalized
    integral.

    """
def optimal_alignment_in_Euclidean_metric(avec, bvec, t=None):
    '''Return rotor R such that R*b⃗*R̄ is as close to a⃗ as possible

    As in the `optimal_alignment_in_chordal_metric` function, the `t` argument is
    optional.  If it is present, the times are used to weight the corresponding
    integral.  If it is not present, a simple sum is used instead (which may be
    slightly faster).

    The task of finding `R` is called "Wahba\'s problem"
    <https://en.wikipedia.org/wiki/Wahba%27s_problem>, and has a simple solution
    using eigenvectors.  In their book "Fundamentals of Spacecraft Attitude
    Determination and Control" (2014), Markley and Crassidis say that "Davenport’s
    method remains the best method for solving Wahba’s problem".  This constructs a
    simple matrix from a sum over the input vectors, and extracts the optimal rotor
    as the dominant eigenvector (the one with the largest eigenvalue).

    '''
def mean_rotor_in_intrinsic_metric(R, t=None) -> None: ...
