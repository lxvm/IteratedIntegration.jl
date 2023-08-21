# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_auxquadgk(f::F, s, n, atol, rtol, maxevals, nrm, segbuf) where {F}
    x,w,gw = cachedrule(eltype(s),n)
    # unlike quadgk, s may be a ntuple or a vector and we try to maintain optimizations
    N = length(s)
    @assert N ≥ 2
    if f isa BatchIntegrand
        segs = evalrules(f, s, x,w,gw, nrm)
    else
        segs = evalrules(f, s, x,w,gw, nrm)
    end
    I, E = resum(f, segs)
    numevals = (2n+1) * (N-1)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E) # fast return when no subdivisions required
    end

    segheap = segbuf === nothing ? collect(segs) : (resize!(segbuf, N-1) .= segs)
    for ord in eachorder(I)
        heapify!(segheap, ord)
        segheap = auxadapt(f, segheap, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm, ord)
        I, E = resum(f, segheap)
        (E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals) && break
    end
    return (I, E)
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function auxadapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ord) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while (tol = max(atol, rtol*nrm(I)); Base.Order.lt(ord, E, tol)) && numevals < maxevals
        next = auxrefine(f, segs, I, E, numevals, x,w,gw,n, tol, atol, rtol, maxevals, nrm, ord)
        next isa Vector && return next # handle type-unstable functions
        I, E, numevals = next
    end
    return segs
end

# internal routine to refine the segment with largest error
function auxrefine(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, tol, atol, rtol, maxevals, nrm, ord) where {F, T}
    s = heappop!(segs, ord)
    mid = (s.a + s.b) / 2
    s1 = evalrule(f, s.a, mid, x,w,gw, nrm)
    s2 = evalrule(f, mid, s.b, x,w,gw, nrm)
    if f isa InplaceIntegrand
        I .= (I .- s.I) .+ s1.I .+ s2.I
    else
        I = (I - s.I) + s1.I + s2.I
    end
    E = (E - s.E) + s1.E + s2.E
    numevals += 4n+2

    # handle type-unstable functions by converting to a wider type if needed
    Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
    if Tj !== T
        return auxadapt(f, heappush!(heappush!(Vector{Tj}(segs), s1, ord), s2, ord),
                     I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, ord)
    end

    # continue bisecting if the remaining error surpasses current tolerance
    tol += s1.E + s2.E
    if Base.Order.lt(ord, E, tol)
        next = auxrefine(f, segs, I,E, numevals, x, w,gw,n, tol, atol, rtol, maxevals, nrm, ord)
        next isa Vector && return heappush!(heappush!(next, s1, ord), s2, ord)
        I, E, numevals = next
    end

    heappush!(segs, s1, ord)
    heappush!(segs, s2, ord)

    return I, E, numevals
end

# re-sum (paranoia about accumulated roundoff)
function resum(f, segs)
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I .+= segs[i].I
            E += segs[i].E
        end
    else
        I = segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I += segs[i].I
            E += segs[i].E
        end
    end
    return (I, E)
end

# Gauss-Kronrod quadrature of f from a to b to c...

"""
    auxquadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm, segbuf=nothing)

Numerically integrate the function `f(x)` from `a` to `b`, and optionally over additional
intervals `b` to `c` and so on. Keyword options include a relative error tolerance `rtol`
(if `atol==0`, defaults to `sqrt(eps)` in the precision of the endpoints), an absolute error tolerance
`atol` (defaults to 0), a maximum number of function evaluations `maxevals` (defaults to
`10^7`), and the `order` of the integration rule (defaults to 7).

Returns a pair `(I,E)` of the estimated integral `I` and an estimated upper bound on the
absolute error `E`. If `maxevals` is not exceeded then `E <= max(atol, rtol*norm(I))`
will hold. (Note that it is useful to specify a positive `atol` in cases where `norm(I)`
may be zero.)

Compared to `quadgk` from QuadGK.jl, `auxquadgk` implements a few more safeguards for
integration of difficult functions. It changes how adaptive refinement is done when using a
relative tolerance to refine all segments with an error above the tolerance (instead of just
the segment with the largest error). Additionally, if an integrand returns an `AuxValue`
then the heap first integrates the auxiliary value followed by the primary by resorting the
heap of segments.

The endpoints `a` et cetera can also be complex (in which case the integral is performed over
straight-line segments in the complex plane). If the endpoints are `BigFloat`, then the
integration will be performed in `BigFloat` precision as well.

!!! note
    It is advisable to increase the integration `order` in rough proportion to the
    precision, for smooth integrands.

More generally, the precision is set by the precision of the integration
endpoints (promoted to floating-point types).

The integrand `f(x)` can return any numeric scalar, vector, or matrix type, or in fact any
type supporting `+`, `-`, multiplication by real values, and a `norm` (i.e., any normed
vector space). Alternatively, a different norm can be specified by passing a `norm`-like
function as the `norm` keyword argument (which defaults to `norm`).

!!! note
    Only one-dimensional integrals are provided by this function. For multi-dimensional
    integration (cubature), there are many different algorithms (often much better than simple
    nested 1d integrals) and the optimal choice tends to be very problem-dependent. See the
    Julia external-package listing for available algorithms for multidimensional integration or
    other specialized tasks (such as integrals of highly oscillatory or singular functions).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each
interval is estimated using a Kronrod rule (`2*order+1` points) and the error is estimated
using an embedded Gauss rule (`order` points). The interval with the largest error is then
subdivided into two intervals and the process is repeated until the desired error tolerance
is achieved.

These quadrature rules work best for smooth functions within each interval, so if your
function has a known discontinuity or other singularity, it is best to subdivide your
interval to put the singularity at an endpoint. For example, if `f` has a discontinuity at
`x=0.7` and you want to integrate from 0 to 1, you should use `auxquadgk(f, 0,0.7,1)` to
subdivide the interval at the point of discontinuity. The integrand is never evaluated
exactly at the endpoints of the intervals, so it is possible to integrate functions that
diverge at the endpoints as long as the singularity is integrable (for example, a `log(x)`
or `1/sqrt(x)` singularity).

For real-valued endpoints, the starting and/or ending points may be infinite. (A coordinate
transformation is performed internally to map the infinite interval to a finite one.)

In normal usage, `auxquadgk(...)` will allocate a buffer for segments. You can
instead pass a preallocated buffer allocated using `alloc_segbuf(...)` as the
`segbuf` argument. This buffer can be used across multiple calls to avoid
repeated allocation.
"""
auxquadgk(f, segs...; kws...) =
    auxquadgk(f, promote(segs...); kws...)

function auxquadgk(f, segs;
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing)
    handle_infinities(f, segs) do g, s, _
        do_auxquadgk(g, s, order, atol, rtol, maxevals, norm, segbuf)
    end
end

"""
    auxquadgk!(f!, result, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm)

Like [`auxquadgk`](@ref), but make use of in-place operations for array-valued integrands (or other mutable
types supporting in-place operations).  In particular, there are two differences from `quadgk`:

1. The function `f!` should be of the form `f!(y, x) = y .= f(x)`.  That is, it writes the
   return value of the integand `f(x)` in-place into its first argument `y`.   (The return
   value of `f!` is ignored.)

2. Like `auxquadgk`, the return value is a tuple `(I,E)` of the estimated integral `I` and the
   estimated error `E`.   However, in `auxquadgk!` the estimated integral is written in-place
   into the `result` argument, so that `I === result`.

Otherwise, the behavior is identical to `auxquadgk`.

For integrands whose values are *small* arrays whose length is known at compile-time,
it is usually more efficient to use `quadgk` and modify your integrand to return
an `SVector` from the [StaticArrays.jl package](https://github.com/JuliaArrays/StaticArrays.jl).
"""
auxquadgk!(f!, result, segs...; kws...) =
    auxquadgk!(f!, result, promote(segs...); kws...)

function auxquadgk!(f!, result, segs; atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing)
    fx = result / oneunit(eltype(segs)) # pre-allocate array of correct type for integrand evaluations
    f = InplaceIntegrand(f!, result, fx)
    return auxquadgk(f, segs; atol=atol, rtol=rtol, maxevals=maxevals, order=order, norm=norm, segbuf=segbuf)
end
