# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_meroquadgk(f::F, s, n, atol, rtol, maxevals, nrm, segbuf, rho, rootmeth) where {F}
    x,w,gw = cachedrule(eltype(s),n)
    fac = cachedlu(eltype(s),n)
    N = length(s)
    @assert N ≥ 2
    fx = Vector{ComplexF64}(undef, 2n+1) # this routine designed for scalar functions
    gx = similar(fx)
    segs = ntuple(Val{N-1}()) do i
        a, b = s[i], s[i+1]
        eval!(fx, f, a, b, x)
        gx .= inv.(fx)
        applypolesub!(gx,fx,a,b,x,w,gw,n,rho,fac,rootmeth,nrm)
    end
    I = sum(s -> s.I, segs)
    E = sum(s -> s.E, segs)
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
    heapify!(segheap, Reverse)
    finalheap = meroadapt(gx, fx, f, segheap, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm, rho,fac,rootmeth)
    return sum(s -> s.I, finalheap), sum(s -> s.E, finalheap)
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function meroadapt(gx, fx, f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, rho,fac,rootmeth) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        next = merorefine(gx, fx, f, segs, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, rho,fac,rootmeth)
        next isa Vector && return next # handle type-unstable functions
        I, E, numevals = next
    end
    return segs
end

# internal routine to refine the segment with largest error
function merorefine(gx, fx, f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm, rho,fac,rootmeth) where {F, T}
    s = heappop!(segs, Reverse)
    mid = (s.a + s.b) / 2
    eval!(fx, f, s.a, mid, x)
    gx .= inv.(fx)
    s1 = applypolesub!(gx,fx,s.a,mid,x,w,gw,n,rho,fac,rootmeth,nrm)
    eval!(fx, f, mid, s.b, x)
    gx .= inv.(fx)
    s2 = applypolesub!(gx,fx,mid,s.b,x,w,gw,n,rho,fac,rootmeth,nrm)
    I = (I - s.I) + s1.I + s2.I
    E = (E - s.E) + s1.E + s2.E
    numevals += 4n+2

    heappush!(segs, s1, Reverse)
    heappush!(segs, s2, Reverse)

    return I, E, numevals
end


# Gauss-Kronrod quadrature of f from a to b to c...

"""
    meroquadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm, segbuf=nothing)

Numerically integrate the function `f(x)` from `a` to `b`, and optionally over additional
intervals `b` to `c` and so on. Keyword options include a relative error tolerance `rtol`
(if `atol==0`, defaults to `sqrt(eps)` in the precision of the endpoints), an absolute error tolerance
`atol` (defaults to 0), a maximum number of function evaluations `maxevals` (defaults to
`10^7`), and the `order` of the integration rule (defaults to 7).

Returns a pair `(I,E)` of the estimated integral `I` and an estimated upper bound on the
absolute error `E`. If `maxevals` is not exceeded then `E <= max(atol, rtol*norm(I))`
will hold. (Note that it is useful to specify a positive `atol` in cases where `norm(I)`
may be zero.)

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
`x=0.7` and you want to integrate from 0 to 1, you should use `meroquadgk(f, 0,0.7,1)` to
subdivide the interval at the point of discontinuity. The integrand is never evaluated
exactly at the endpoints of the intervals, so it is possible to integrate functions that
diverge at the endpoints as long as the singularity is integrable (for example, a `log(x)`
or `1/sqrt(x)` singularity).

For real-valued endpoints, the starting and/or ending points may be infinite. (A coordinate
transformation is performed internally to map the infinite interval to a finite one.)

In normal usage, `meroquadgk(...)` will allocate a buffer for segments. You can
instead pass a preallocated buffer allocated using `alloc_segbuf(...)` as the
`segbuf` argument. This buffer can be used across multiple calls to avoid
repeated allocation.
"""
meroquadgk(f, segs...; kws...) =
    meroquadgk(f, promote(segs...); kws...)

function meroquadgk(f, segs::T;
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing, rho=1.0, rootmeth=NewtonDeflation()) where {T}
    handle_infinities(f, segs) do f, s, _
        do_meroquadgk(f, s, order, atol, rtol, maxevals, norm, segbuf, rho, rootmeth)
    end
end
