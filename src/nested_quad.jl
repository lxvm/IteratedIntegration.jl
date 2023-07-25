measure(segs) = abs(segs[end]-segs[1])
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
struct NestedGaussKronrod{dim,d,X}
    x::SVector{d,X}
    function NestedGaussKronrod{dim}(x::SVector{d,X}) where {dim,d,X}
        return new{dim,d,X}(x)
    end
end
function NestedGaussKronrod(T,::Val{d}) where {d}
    X = float(T)
    vec= SVector{0,X}()
    return NestedGaussKronrod{d}(vec)
end

# stuff for 1d quadrature
rule_type(::NestedGaussKronrod{dim,d,X}) where {dim,d,X} = SVector{dim+d,X}
# base case
rule_finalize(f::NestedGaussKronrod, x) = pushfirst(f.x, x)

# multidimensional
nextdim(::NestedGaussKronrod{dim}) where {dim} = dim
function nextrule(g::NestedGaussKronrod{dim}, x, ::Val{dim}) where {dim}
    return NestedGaussKronrod{dim-1}(pushfirst(g.x, x))
end

#=
# dim stores the active/current dimension
struct QuadNest{d,FW,F,L,R,A,Z,N,I,S}
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
end

function (q::QuadNest{dim})(x) where {dim}
    ndims(q.l) === 1 && return q.f(rule_finalize(q.rule, x))
    rule = nextrule(q.rule, x, Val(dim))
    l = fixandeliminate(q.l, x, Val(dim))
    I, = do_nested_quad(Val(ndims(l)), q.fw, q.f, l, rule, q.atol, q.rtol, q.maxevals, q.norm, q.initdivs, q.segbufs)
    return I
end

function do_nested_quad(::Val{d}, fw::Val{FW}, f, l, rule, atol, rtol, maxevals, nrm, initdivs, segbufs) where {d,FW}
    dim = nextdim(rule)
    segs = initsegs(segments(l, dim), initdivs[dim])
    initdivs_= restof(Val(dim), initdivs)
    segbufs_ = restof(Val(dim), segbufs)
    q = QuadNest(Val(dim), fw, f, l, rule, nextatol(atol, segs), rtol, maxevals, nrm, initdivs_, segbufs_)
    fun = FW(q) # we put the FunctionWrapper here since this leads to much less specialization
    return auxquadgk(fun, segs..., atol=atol, rtol=rtol, maxevals=maxevals, norm=nrm, segbuf=segbufs[dim])
end
=#

# dim stores the active/current dimension
struct QuadNest{d,FW,F,L,R,A,Z,N,I,S}
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
end


restof_(::Val) = throw(ArgumentError("Empty tuple"))
restof_(::Val{1}, _, args...) = args
restof_(::Val{d}, x, args...) where {d} = (x, restof_(Val(d-1), args...)...)
restof(valdim::V, t::Tuple) where {V} = restof_(valdim, t...)

function make_nest(f, l, ::Nothing)
    return QuadNest(f, fw, )
end
function make_nest(f, l, (dim, state))
    next = iterate(l, state)
    restof(dim, initdivs)
    q = make_nest(f, l, next)
    return QuadNest(q, l, fw)
end

function (q::QuadNest)(x)
    if ndims(q.l) === 1
        return q.f(rule_finalize(q.rule, x))
    else
    dim = nextdim(q.l)
    segs = segments(q.l, dim)
    nest = nextnest(q, x, Val(dim))
    g = nest.fw(nest)
    I, = auxquadgk(g, segs..., atol=atol, rtol=rtol, maxevals=maxevals, norm=nrm, segbuf=segbufs[dim])
    return I
end


"""
    nested_quadgk(f, a, b; kwargs...)
    nested_quadgk(f, l::AbstractIteratedLimits{d,T}; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=typemax(Int), initdivs=ntuple(i -> Val(1), Val{d}()), segbufs=nothing, rule=nothing) where {d,T}

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

The `initdivs` keyword allows passing a tuple of integers which specifies the initial number
of panels in each `quadgk` call at each level of integration.

In normal usage, `nested_quadgk` will allocate segment buffers. You can instead pass a
preallocated buffer allocated using [`alloc_segbufs`](@ref) as the segbuf argument. This
buffer can be used across multiple calls to avoid repeated allocation.
"""
nested_quadgk(f, a, b; kwargs...) = nested_quad(f, CubicLimits(a, b); kwargs...)
function nested_quadgk(f::F, l::L; order=7, atol=nothing, rtol=nothing, norm=norm, maxevals=10^7, initdivs=nothing, segbufs=nothing, nest=nothing) where {F,d,T,L<:AbstractIteratedLimits{d,T}}
    initdivs_ = initdivs === nothing ? ntuple(i -> Val(1), Val(d)) : initdivs
    segbufs_ = segbufs === nothing ? ntuple(i -> nothing, Val(d)) : segbufs
    nest_ = nest === nothing ? make_nest(f, l, iterate(l), order, atol_, rtol_, initdivs_, segbufs_) : nest
    dim = nextdim(l)
    segs= segments(l, dim)
    g = nest_.fw(nest_)
    return auxquadgk(g, segs..., atol=atol, rtol=rtol, maxevals=maxevals, norm=norm, initdiv=initdivs_[dim], segbuf=segbufs_[dim])
    # do_nested_quad(Val(d), Val(FW), f, l, rule_, atol_, rtol_, maxevals, norm, initdivs_, segbufs_)
end

"""
    alloc_segbufs(ndim, [coefficient_type, range_type, norm_type])

This helper function will allocate enough segment buffers as are needed for a
`nested_quadgk` call of integrand `f` and integration limits `l`.
`coefficient_type` should be `eltype(l)`, `typesof_fx` should be the return type of the
integrand `f` for each iteration of integration, `typesof_nfx` should be the
types of the norms of a value of `f` for each iteration of integration, and
`ndim` should be `ndims(l)`.
"""
function alloc_segbufs(ndim::Integer, args...; kwargs...)
    return ntuple(n -> alloc_segbuf(args...; kwargs...), ndim)
end


# this represents multiple product rules
# this time rules represents the active rules and vals represents the suspended rules
# we have to trick the rules into not seeing the base case
struct NestedProductRule{dim,R,V}
    rules::R
    vals::V
    NestedProductRule{dim}(r::R, v::V) where {dim,R,V} = new{dim,R,V}(r, v)
end
# a product rule passes productvalues to its integrand

# a container to pass to integrand
struct ProductValue{T<:Tuple}
    v::T
end

function Base.zero(::Type{<:ProductValue{T}}) where {T<:Tuple}
    vals = ntuple(n -> zero(fieldtype(T, n)), Val(fieldcount(T)))
    return ProductValue(vals)
end

# base case
function rule_type(g::NestedProductRule)
    return ProductValue{Tuple{map(r -> rule_type(r), g.rules)..., map(typeof, g.vals)...}}
end
function (g::NestedProductRule{1})(f::F, s, nrm=norm, buffer=nothing) where {F}
    return first(g.rules)(x -> f(ProductValue((x, g.vals...))), s, nrm, buffer)
end

# multidimensional
Base.ndims(::NestedProductRule{dim}) where {dim} = dim
Base.ndims(::NestedGaussKronrod{dim}) where {dim} = dim

nextproddim_() = 0
nextproddim_(r, rules...) = ndims(r) + nextproddim_(rules...)
nextproddim(r, rules...)  = nextdim(r) + nextproddim_(rules...)
nextdim(r::NestedProductRule) = nextproddim(r.rules...)

# this is a redefinition of the base case
complete_rule(rule::NestedGaussKronrod, x) = pushfirst(rule.x, x)
function nextprodrule(x, ::Val{d}, r, rules...) where {d}
    if ndims(r) === 1
        # reached the base case of the current rule
        return rules, (complete_rule(r, x),)
    else
        # reduce the current rule
        nr = nextrule(r, x, Val(d-nextproddim_(rules...)))
        return (nr, rules...), ()
    end
end
function nextrule(r::NestedProductRule{dim}, x, v::Val{d}) where {dim,d}
    nr, nv = nextprodrule(x, v, r.rules...)
    return NestedProductRule{dim-1}(nr, (nv..., r.vals...))
end

# here we elide the base case
function apply_multidim(rule::NestedGaussKronrod, f::F, s, nrm=norm, buffer=nothing) where {F}
    return rule.g(f, s, nrm, buffer)
end
function (g::NestedProductRule)(f::F, s, nrm=norm, buffer=nothing) where {F}
    return apply_multidim(first(g.rules), f, s, nrm, buffer)
end
