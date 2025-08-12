"""
    CubicLimits(a, b)

Store integration limit information for a hypercube with vertices `a` and `b`.
which can be can be real numbers, tuples, or `AbstractVector`s.
The outermost variable of integration corresponds to the last entry.
"""
struct CubicLimits{d,T} <: AbstractIteratedLimits{d,T}
    a::SVector{d,T}
    b::SVector{d,T}
end
function CubicLimits(a::SVector{d,A}, b::SVector{d,B}) where {d,A<:Number,B<:Number}
    T = promote_type(A, B)
    return CubicLimits{d,T}(convert(SVector{d,T}, a), convert(SVector{d,T},b))
end
function CubicLimits(a::NTuple{d}, b::NTuple{d}) where {d}
    return CubicLimits(SVector{d,eltype(a)}(a), SVector{d,eltype(b)}(b))
end
CubicLimits(::Tuple{}, ::Tuple{}) = error("0-d cube not possible")
CubicLimits(a, b) = CubicLimits(promote(a...), promote(b...))

function fixandeliminate(c::CubicLimits{d,T}, _, ::Val{dim}) where {d,T,dim}
    return CubicLimits{d-1,T}(deleteat(c.a, dim), deleteat(c.b, dim))
end
function segments(c::CubicLimits, dim)
    return (c.a[dim], c.b[dim])
end
function eliminate(c::CubicLimits{d,T}, dims) where {d,T}
    length(dims) == d && return nothing
    dimdiff = setdiff(1:d, dims)
    return CubicLimits{d-length(dims),T}(c.a[dimdiff], c.b[dimdiff])
end

# TODO remove the s variable in TetrahedralLimits and just store the data of the tetrahedron
"""
    TetrahedralLimits(a::NTuple{d}) where d

A parametrization of the integration limits for a tetrahedron whose vertices are
```
( 0.0,  0.0, ...,  0.0)
( 0.0,  0.0, ..., a[d])
…
( 0.0, a[2], ..., a[d])
(a[1], a[2], ..., a[d])
```
"""
struct TetrahedralLimits{d,T,A} <: AbstractIteratedLimits{d,T}
    a::A
    TetrahedralLimits(::Tuple{}) = error("0-d simplex not possible")
    TetrahedralLimits(a::Tuple{T,Vararg{T,n}}) where {n,T} = new{n+1,T,typeof(a)}(a)
end
# TetrahedralLimits(a::Tuple) = TetrahedralLimits(promote(a...))
TetrahedralLimits(a::AbstractVector) = TetrahedralLimits(promote(a...))

function fixandeliminate(t::TetrahedralLimits{d,T}, x, ::Val{d}) where {d,T}
    _x = convert(float(T), x)
    # @assert zero(_x) <= _x <= t.a[d] " 0 <= $_x <= $(t.a[d])"
    s = iszero(t.a[d]) ? one(_x*t.a[d]) : _x/t.a[d]
    return TetrahedralLimits(Base.front(t.a) .* s)
end
function segments(t::TetrahedralLimits{d,T}, dim) where {d,T}
    return (zero(T), t.a[dim])
end
function eliminate(t::TetrahedralLimits{d,T}, dims) where {d,T}
    length(dims) == d && return nothing
    dimdiff = setdiff(1:d, dims)
    return TetrahedralLimits(t.a[dimdiff])
end


"""
    ProductLimits(lims::AbstractIteratedLimits...)

Construct a collection of limits which yields the first limit followed by the
second, and so on. The inner limits are not allowed to depend on the outer ones.
The outermost variable of integration should be placed first, i.e.
``\\int_{\\Omega} \\int_{\\Gamma}`` should be `ProductLimits(Ω, Γ)`.
Although changing the order of the limits should not change the results, it may affect the efficiency of integration.
"""
struct ProductLimits{d,T,L} <: AbstractIteratedLimits{d,T}
    lims::L
    ProductLimits{d,T}(lims::L) where {d,T,L<:Tuple{Vararg{AbstractIteratedLimits}}} =
        new{d,T,L}(lims)
end
ProductLimits(lims::AbstractIteratedLimits...) = ProductLimits(lims)
function ProductLimits(lims::L) where {L<:Tuple{Vararg{AbstractIteratedLimits}}}
    d = mapreduce(ndims, +, lims; init=0)
    T = mapreduce(eltype, promote_type, lims)
    ProductLimits{d,T}(lims)
end

nextprodlim(x, lim, lims...) = (fixandeliminate(lim, x, Val(ndims(lim))), lims...)
# after exhausting the active limits, move onto the rest
nextprodlim(_, ::AbstractIteratedLimits{1}, lims...) = lims
function fixandeliminate(l::ProductLimits{d,T}, x, ::Val{d}) where {d,T}
    return ProductLimits{d-1,T}(nextprodlim(x, l.lims...))
end

function segments(l::ProductLimits{d}, dim) where {d}
    @assert 0<dim<=d
    rem = d-ndims(l.lims[1])
    if dim > rem
        segments(l.lims[1], dim-rem)
    else
        return segments(ProductLimits(l.lims[2:end]...), dim)
    end
end

function eliminate(l::ProductLimits{d}, dim::Int) where {d}
    @assert 0<dim<=d
    rem = d-ndims(l.lims[1])
    if dim > rem
        out = eliminate(l.lims[1], dim-rem)
        out === nothing && return (l.lims[2:end] isa Tuple{} ? nothing : ProductLimits(l.lims[2:end]...))
        return ProductLimits(out, l.lims[2:end]...)
    else
        tmp = eliminate(ProductLimits(l.lims[2:end]...), dim)
        tmp === nothing && return ProductLimits(l.lims[1])
        return ProductLimits(l.lims[1], tmp.lims...)
    end
end
function eliminate(l::ProductLimits, dims)
    _dims = sort(dims)
    for d in reverse(_dims)
        l = eliminate(l, d)
    end
    return l
end


"""
    TranslatedLimits(lims::AbstractIteratedLimits{d}, t::NTuple{d}) where d

Returns the limits of `lims` translated by offsets in `t`.
"""
struct TranslatedLimits{d,C,L,T} <: AbstractIteratedLimits{d,C}
    l::L
    t::T
    TranslatedLimits{d,C}(l::L, t::T) where {d,C,L<:AbstractIteratedLimits{d,C},T<:NTuple{d}} =
        new{d,C,L,T}(l, t)
end
function TranslatedLimits(l::AbstractIteratedLimits{d,C}, t::NTuple{d}) where {d,C}
    return TranslatedLimits{d,C}(l, t)
end

function fixandeliminate(t::TranslatedLimits{d,C}, x, ::Val{dim}) where {d,C,dim}
    l = fixandeliminate(t.l, x - t.t[ndims(t)], Val(dim))
    return TranslatedLimits{d-1,eltype(l)}(l, Base.front(t.t))
end
function segments(t::TranslatedLimits, dim)
    return map(x -> convert(eltype(t), x) + t.t[dim], segments(t.l, dim))
end
function eliminate(t::TranslatedLimits, dims)
    out = eliminate(t.l, dims)
    out === nothing && return nothing
    return TranslatedLimits(out, t.t[setdiff(1:ndims(t), dims)])
end

# More ideas for limits
# ParametrizedLimits
# requiring linear programming
# RotatedLimits
# AffineLimits

