measure(segs) = sum(s -> abs(s[2]-s[1]), segs)
nextatol(atol, segs) = atol/measure(segs)

initsegs(segs, ::Val{1}) = segs

function initsegs_(a, b, initdivs)
    r = range(a, b, length=initdivs+1)
    ntuple(i -> r[i], Val(initdivs+1))
end
# function initsegs(segs::NTuple{N,NTuple{2,T}}, ::Val{initdiv}) where {T,initdiv}
#     # TODO, split each segment into initdiv parts # we need to flatten the split segment
#     iterator and return something with an eltype, length (and indexable for parallelization)
#     return
# end

# dim represents the active/next dimension and d the total dimension
# we traverse the dimensions from d:-1:1 i.e. total remaining dimension
struct NestedGaussKronrod{dim,d,X,T}
    x::SVector{d,X}
    g::RuleQuad.GaussKronrod{T}
    function NestedGaussKronrod{dim}(x::SVector{d,X}, g::RuleQuad.GaussKronrod{T}) where {dim,d,X,T}
        return new{dim,d,X,T}(x, g)
    end
end
function NestedGaussKronrod(T,n,::Val{d}) where {d}
    X = float(T)
    vec= SVector{0,X}()
    gk = RuleQuad.GaussKronrod(T,n)
    return NestedGaussKronrod{d}(vec,gk)
end

# stuff for 1d quadrature
countevals(g::NestedGaussKronrod) = countevals(g.g)
rule_type(::NestedGaussKronrod{dim,d,X,T}) where {dim,d,X,T} = SVector{dim+d,X}
# base case
function (g::NestedGaussKronrod{1})(f::F, s, nrm=norm, buffer=nothing) where {F}
    return g.g(x -> f(pushfirst(g.x, x)), s, nrm, buffer)
end
# the signal to do the innermost integral is when the total remaining dimension is 1
function do_nested_quad(::Val{1}, FW::Val, f, l, rule, atol, rtol, maxevals, nrm, initdivs, segbufs, parallels)
    segs = initsegs(segments(l, 1), initdivs[1])
    return RuleQuad.do_auxquad(f, segs, rule, atol, rtol, maxevals, nrm, segbufs[1], parallels[1])
end

# dim stores the active/current dimension
struct QuadNest{d,FW,F,L,R,A,Z,N,I,S,P}
    dim::Val{d}
    fw::Val{FW}
    f::F
    l::L
    rule::R
    atol::A
    rtol::Z
    maxevals::Int
    norm::N
    initdivs::I
    segbufs::S
    parallels::P
end

# multidimensional
@inline nextdim(::NestedGaussKronrod{dim}) where {dim} = dim

function (g::NestedGaussKronrod)(f::F, s, nrm=norm, buffer=nothing) where {F}
    return g.g(f, s, nrm, buffer)
end

function nextrule(g::NestedGaussKronrod{dim}, x, ::Val{dim}) where {dim}
    return NestedGaussKronrod{dim-1}(pushfirst(g.x, x), g.g)
end

restof_(::Val) = throw(ArgumentError("Empty tuple"))
restof_(::Val{1}, _, args...) = args
restof_(::Val{d}, x, args...) where {d} = (x, restof_(Val(d-1), args...)...)
restof(valdim::V, t::Tuple) where {V} = restof_(valdim, t...)

function (q::QuadNest{dim})(x) where {dim}
    rule = nextrule(q.rule, x, Val(dim))
    l = fixandeliminate(q.l, x, Val(dim))
    I, = do_nested_quad(Val(ndims(l)), q.fw, q.f, l, rule, q.atol, q.rtol, q.maxevals, q.norm, q.initdivs, q.segbufs, q.parallels)
    return I
end

function do_nested_quad(::Val{d}, fw::Val{FW}, f, l, rule, atol, rtol, maxevals, nrm, initdivs, segbufs, parallels) where {d,FW}
    dim = nextdim(rule)
    segs = initsegs(segments(l, dim), initdivs[dim])
    initdivs_= restof(Val(dim), initdivs)
    segbufs_ = restof(Val(dim), segbufs)
    parallels_ = restof(Val(dim), parallels)
    q = QuadNest(Val(dim), fw, f, l, rule, nextatol(atol, segs), rtol, maxevals, nrm, initdivs_, segbufs_, parallels_)
    fun = FW(q) # we put the FunctionWrapper here since this leads to much less specialization
    return RuleQuad.do_auxquad(fun, segs, rule, atol, rtol, maxevals, nrm, segbufs[dim], parallels[dim])
end

"""
    nested_quad(f, a, b; kwargs...)
    nested_quad(f, l::AbstractIteratedLimits{d,T}; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing, parallels=nothing, rule=NestedGaussKronrod) where {d,T}

Calls `QuadGK` to perform iterated 1D integration of `f` over a compact domain parametrized
by `AbstractIteratedLimits` `l`. In the case two points `a` and `b` are passed, the
integration region becomes the hypercube with those extremal vertices (which mimics
`hcubature`).

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
nested_quad(f, a, b; kwargs...) = nested_quad(f, CubicLimits(a, b); kwargs...)
function nested_quad(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=nothing, segbufs=nothing, parallels=nothing, rule=nothing, RT=nothing) where {F,d,T,L<:AbstractIteratedLimits{d,T}}
    initdivs_ = initdivs === nothing ? ntuple(i -> Val(1), Val(d)) : initdivs
    segbufs_ = segbufs === nothing ? ntuple(i -> nothing, Val(d)) : segbufs
    parallels_ = parallels === nothing ? ntuple(i -> Sequential(), Val(d)) : parallels
    rule_ = rule === nothing ? NestedGaussKronrod(T, order, Val(d)) : rule
    TF = typeof(float(real(one(T))))
    atol_ = something(atol, zero(TF))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(TF))) : zero(TF))
    RT_ = RT === nothing ? typeof(f(zero(rule_type(rule_)))) : RT
    FW = FunctionWrapper{RT_, Tuple{float(T)}}
    do_nested_quad(Val(d), Val(FW), f, l, rule_, atol_, rtol_, maxevals, norm, initdivs_, segbufs_, parallels_)
end

"""
    alloc_segbufs(coefficient_type, range_type, norm_type, ndim)

This helper function will allocate enough segment buffers as are needed for an
`nested_quadgk` call of integrand `f` and integration limits `l`.
`coefficient_type` should be `eltype(l)`, `typesof_fx` should be the return type of the
integrand `f` for each iteration of integration, `typesof_nfx` should be the
types of the norms of a value of `f` for each iteration of integration, and
`ndim` should be `ndims(l)`.
"""
alloc_segbufs(coefficient_type, range_type, norm_type, ndim::Int) =
    ntuple(n -> alloc_segbuf(coefficient_type, range_type, norm_type), ndim)
