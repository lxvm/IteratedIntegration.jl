iterated_inner_tol(atol, a, b) = atol/(b - a)
iterated_outer_tol(atol, a, b) = atol

"""
    iterated_inference(F, T, ::Val{d}) where d
    iterated_inference(f, l::AbstractIteratedLimits)

Returns a tuple of the return types of f after each variable of integration
"""
iterated_inference(F, T, vald) =
    iterated_inference_up(iterated_inference_down(F, T, vald), T, vald)
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
    iterated_segs(f, l, a, b, ::Val{initdivs}) where initdivs

Returns a `Tuple` of integration nodes that are passed to `QuadGK` to initialize
the segments for adaptive integration. By default, returns `initdivs` equally
spaced panels on interval `(a, b)`. If `f` is localized, specializing this
function can also help avoid errors when `QuadGK` fails to adapt.
"""
function iterated_segs(_, _, a, b, ::Val{initdivs}) where initdivs
    r = range(a, b, length=initdivs+1)
    ntuple(i -> r[i], Val(initdivs+1))
end


struct QuadNest{d,F,L,T,A,R,N,I,S,U,Y}
    D::Val{d}
    f::F
    l::L
    a::T
    b::T
    order::Int
    atol::A
    rtol::R
    maxevals::Int
    norm::N
    initdivs::I
    segbufs::S
    routine::U
    types::Y
end

function do_nested_quadgk(q::QuadNest{1})
    segs = iterated_segs(q.f, q.l, q.a, q.b, q.initdivs[1])
    do_quadgk(q.f, segs, q.order, q.atol, q.rtol, q.maxevals, q.norm, q.segbufs[1])
end

function (q::QuadNest{d})(x) where d
    atol = iterated_inner_tol(q.atol, q.a, q.b)
    g = iterated_pre_eval(q.f, x, q.D)
    m = fixandeliminate(q.l, x)
    a, b = endpoints(m)
    p = QuadNest(Val(d-1), g, m,a,b, q.order, atol, q.rtol, q.maxevals, q.norm, q.initdivs[1:d-1], q.segbufs[1:d-1], q.routine, q.types)
    I, = q.routine(p)
    iterated_integrand(q.f, I, q.D)
end

function do_nested_quadgk(q::QuadNest{d,F,L,T}) where {d,F,L,T}
    segs = iterated_segs(q.f, q.l, q.a, q.b, q.initdivs[d])
    atol = iterated_outer_tol(q.atol, q.a, q.b)
    func = FunctionWrapper{q.types[d],Tuple{T}}(q)
    do_quadgk(func, segs, q.order, atol, q.rtol, q.maxevals, q.norm, q.segbufs[d])
end

"""
    nested_quadgk(f, a, b; kwargs...)
    nested_quadgk(f::Function, l; kwargs...)
    nested_quadgk(f::AbstractIteratedIntegrand{d}, ::AbstractIteratedLimits{d}; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing) where d

Calls `QuadGK` to perform iterated 1D integration of `f` over a compact domain parametrized
by `AbstractIteratedLimits` `l`. In the case two points `a` and `b` are passed, the
integration region becomes the hypercube with those extremal vertices (which mimics
`hcubature`). `f` must implement the [`AbstractIteratedIntegrand`](@ref) interface, or if it
is a `Function` it will be wrapped in a [`ThunkIntegrand`](@ref) so that it is treated like
one. `f` is assumed to be type-stable and its return type is inferred and enforced. Errors
from `iterated_inference` are an indication that inference failed and there could be an
instability or bug with the integrand.

Returns a tuple `(I, E)` of the estimated integral and estimated error.

Keyword options include a relative error tolerance `rtol` (if `atol==0`, defaults to
`sqrt(eps)` in the precision of the norm of the return type), an absolute error tolerance
`atol` (defaults to 0), a maximum number of function evaluations `maxevals` for each nested
integral (defaults to `10^7`), and the `order` of the integration rule (defaults to 7).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each
interval is estimated using a Kronrod rule (`2*order+1` points) and the error is estimated
using an embedded Gauss rule (`order` points). The interval with the largest error is then
subdivided into two intervals and the process is repeated until the desired error tolerance
is achieved. This 1D procedure is applied recursively to each variable of integration in an
order determined by `l` to obtain the multi-dimensional integral.

Unlike `quadgk`, this routine does not allow infinite limits of integration nor unions of
intervals to avoid singular points of the integrand. However, the `initdivs` keyword allows
passing a tuple of integers which specifies the initial number of panels in each `quadgk`
call at each level of integration.

In normal usage, `nested_quadgk` will allocate segment buffers. You can instead pass a
preallocated buffer allocated using [`alloc_segbufs`](@ref) as the segbuf argument. This
buffer can be used across multiple calls to avoid repeated allocation.
"""
nested_quadgk(f, a, b; kwargs...) = nested_quadgk(f, CubicLimits(a, b); kwargs...)
nested_quadgk(f::Function, l; kwargs...) = nested_quadgk(ThunkIntegrand{ndims(l)}(f), l; kwargs...)
function nested_quadgk(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=nothing, segbufs=nothing) where {F,L}
    initdivs_ = initdivs === nothing ? ntuple(i -> Val(1), Val{ndims(l)}()) : initdivs
    segbufs_ = segbufs === nothing ? alloc_segbufs(f, l) : segbufs
    atol_ = something(atol, zero(eltype(l)))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(l)))) : zero(eltype(l)))
    a, b = endpoints(l)
    types = types_of_segbufs(segbufs_, Val(ndims(l)))
    q = QuadNest(Val(ndims(l)), f, l,a,b, order, atol_, rtol_, maxevals, norm, initdivs_, segbufs_, do_nested_quadgk, types)
    do_nested_quadgk(q)
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

type_of_segbuf(::Vector{Segment{TX,TI,TE}}) where {TX,TI,TE} = TI
function types_of_segbufs(segbufs, ::Val{N}) where N
    ntuple(i -> type_of_segbuf(segbufs[i]), Val{N}())
end
