"""
    iterated_tol_update(f, l, atol, rtol)

Choose a new set of error tolerances for the next inner integral. By default
returns `(atol, rtol)` unchanged.
"""
iterated_tol_update(f, l, atol, rtol, dim) = (atol, rtol)

"""
    iterated_inference(F, T, ::Val{d}) where d
    iterated_inference(f, l::AbstractIteratedLimits)

Returns a tuple of the return types of f after each variable of integration
"""
iterated_inference(F, T, ::Val{d}) where d =
    iterated_inference_up(iterated_inference_down(F, T, Val(d)), T, Val(d))
iterated_inference(f, l) =
    iterated_inference(typeof(f), eltype(l), Val(ndims(l)))


function iterated_inference_down(F, T, ::Val{d}) where d
    # recurse down to the innermost integral to get the integrand function types
    Finner = Base.promote_op(iterated_pre_eval, F, T, Val{d})
    (iterated_inference_down(Finner, T, Val(d-1))..., F)
end
iterated_inference_down(F, T, ::Val{1}) = (F,)

function iterated_inference_up(Fs::NTuple{d_}, T, dim::Val{d}) where {d_,d}
    # go up the call stack and get the integrand result types
    Fouter = Base.promote_op(iterated_integrand, Fs[1], T, Val{d-d_+1}) # output type
    Fouter === Union{} && error("Could not infer the output type of the integrand. Check that it runs and is type stable")
    (Fouter, iterated_inference_up(Fs[2:d_], Fouter, dim)...)
end
iterated_inference_up(Fs::NTuple{1}, T, ::Val{d}) where d =
    (Base.promote_op(iterated_integrand, Fs[1], T, Val{d}),)

"""
    iterated_integral_type(f, l::AbstractIteratedLimits)

Returns the output type of `nested_quadgk(f, l)`
"""
function iterated_integral_type(f, l)
    T = iterated_inference(f, l)[ndims(l)]
    Tuple{T,Base.promote_op(norm, T)}
end

"""
    iterated_segs(f, l::AbstractIteratedLimits, ::Val{initdivs}) where initdivs

Returns a `Tuple` of integration nodes that are passed to `QuadGK` to initialize
the segments for adaptive integration. By default, returns `initdivs` equally
spaced panels on `endpoints(l)`. If `f` is localized, specializing this
function can also help avoid errors when `QuadGK` fails to adapt.
"""
function iterated_segs(_, l, ::Val{initdivs}) where initdivs
    a, b = endpoints(l)
    r = range(a, b, length=initdivs+1)
    ntuple(i -> r[i], Val(initdivs+1))
end


struct QuadNest{d,F,L,A,R,N,I,S}
    D::Val{d}
    f::F
    l::L
    order::Int
    atol::A
    rtol::R
    maxevals::Int
    norm::N
    initdivs::I
    segbufs::S
end

(q::QuadNest{1})(x) = q.f(x)
function (q::QuadNest{d})(x) where d
    g = iterated_pre_eval(q.f, x, q.D)
    m = fixandeliminate(q.l, x)
    atol, rtol = iterated_tol_update(q.f, q.l, q.atol, q.rtol, d)
    p = QuadNest(Val(d-1), g, m, q.order, atol, rtol, q.maxevals, q.norm, q.initdivs, q.segbufs)
    I, = do_quadgk(p, iterated_segs(g, m, q.initdivs[d-1]), q.order, q.atol, q.rtol, q.maxevals, q.norm, q.segbufs[d-1])
    iterated_integrand(g, I, q.D)
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
function nested_quadgk(f, a, b; kwargs...)
    l = CubicLimits(a, b)
    nested_quadgk(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
end
function nested_quadgk(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=nothing, segbufs=nothing) where {F,L}
    initdivs_ = initdivs === nothing ? ntuple(i -> Val(1), Val{ndims(l)}()) : initdivs
    segbufs_ = segbufs === nothing ? alloc_segbufs(f, l) : segbufs
    atol_ = something(atol, zero(eltype(l)))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(l)))) : zero(eltype(l)))
    q = QuadNest(Val(ndims(l)), f, l, order, atol_, rtol_, maxevals, norm, initdivs_, segbufs_)
    do_quadgk(q, iterated_segs(f, l, initdivs_[end]), order, atol_, rtol_, maxevals, norm, segbufs_[end])
end

"""
    alloc_segbufs(coefficient_type, range_type, norm_type, ndim)
    alloc_segbufs(coefficient_type, typesof_fx, typesof_nfx, ndim)
    alloc_segbufs(f, l::AbstractIteratedLimits)

This helper function will allocate enough segment buffers as are needed for an
`nested_quadgk` call of integrand `f` and integration limits `l`.
`coefficient_type` should be `eltype(l)`, `typesof_fx` should be the return type of the
integrand `f` for each iteration of integration, `typesof_nfx` should be the
types of the norms of a value of `f` for each iteration of integration, and
`ndim` should be `ndims(l)`.
"""
alloc_segbufs(coefficient_type, range_type, norm_type, ndim::Int) =
    ntuple(n -> alloc_segbuf(coefficient_type, range_type, norm_type), ndim)
alloc_segbufs(coefficient_type, typesof_fx::Tuple, typesof_nfx::Tuple, ndim::Int) =
    ntuple(n -> alloc_segbuf(coefficient_type, typesof_fx[n], typesof_nfx[n]), ndim)
function alloc_segbufs(f, l)
    typesof_fx = iterated_inference(f, l)
    typesof_nfx = ntuple(n -> Base.promote_op(norm, typesof_fx[n]), Val{ndims(l)}())
    alloc_segbufs(eltype(l), typesof_fx, typesof_nfx, ndims(l))
end
