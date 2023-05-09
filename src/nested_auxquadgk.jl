function do_nested_auxquadgk(q::QuadNest{1})
    segs = iterated_segs(q.f, q.a, q.b, q.initdivs[1])
    do_auxquadgk(q.f, segs, q.order, q.atol, q.rtol, q.maxevals, q.norm, q.segbufs[1])
end

function do_nested_auxquadgk(q::QuadNest{d}) where d
    segs = iterated_segs(q.f, q.a, q.b, q.initdivs[d])
    atol = iterated_outer_tol(q.atol, q.a, q.b)
    do_auxquadgk(q, segs, q.order, atol, q.rtol, q.maxevals, q.norm, q.segbufs[d])
end

"""
    nested_quadgk(f, a, b; kwargs...)
    nested_quadgk(f::AbstractIteratedIntegrand{d}, ::AbstractIteratedLimits{d}; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing) where d

Calls `QuadGK` to perform iterated 1D integration of `f` over a compact domain
parametrized by `AbstractIteratedLimits`. In the case two points `a` and `b` are
passed, the integration region becomes the hypercube with those extremal
vertices (which mimics `hcubature`). `f` is assumed to be type-stable.

Returns a tuple `(I, E)` of the estimated integral and estimated error.

Keyword options include a relative error tolerance `rtol` (if `atol==0`,
defaults to `sqrt(eps)` in the precision of the norm of the return type), an
absolute error tolerance `atol` (defaults to 0), a maximum number of function
evaluations `maxevals` for each nested integral (defaults to `10^7`), and the
`order` of the integration rule (defaults to 7).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral
in each interval is estimated using a Kronrod rule (`2*order+1` points) and the
error is estimated using an embedded Gauss rule (`order` points). The interval
with the largest error is then subdivided into two intervals and the process is
repeated until the desired error tolerance is achieved. This 1D procedure is
applied recursively to each variable of integration in an order determined by
`l` to obtain the multi-dimensional integral.

Unlike `quadgk`, this routine does not allow infinite limits of integration nor
unions of intervals to avoid singular points of the integrand. However, the
`initdivs` keyword allows passing a tuple of integers which specifies the
initial number of panels in each `quadgk` call at each level of integration.

In normal usage, `nested_quadgk` will allocate segment buffers. You can
instead pass a preallocated buffer allocated using [`alloc_segbufs`](@ref) as
the segbuf argument. This buffer can be used across multiple calls to avoid
repeated allocation.
"""
function nested_auxquadgk(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    nested_auxquadgk(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
function nested_auxquadgk(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=nothing, segbufs=nothing) where {F,L}
    initdivs_ = initdivs === nothing ? ntuple(i -> Val(1), Val{ndims(l)}()) : initdivs
    segbufs_ = segbufs === nothing ? alloc_segbufs(f, l) : segbufs
    atol_ = something(atol, zero(eltype(l)))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(l)))) : zero(eltype(l)))
    a, b = endpoints(l)
    q = QuadNest(Val(ndims(l)), f, l,a,b, order, atol_, rtol_, maxevals, norm, initdivs_, segbufs_, do_nested_auxquadgk)
    do_nested_auxquadgk(q)
end

