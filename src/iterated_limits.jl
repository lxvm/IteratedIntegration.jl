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
    s::T
    TetrahedralLimits{d,T}(a::A, s::T) where {d,T,A<:NTuple{d,T}} = new{d,T,A}(a, s)
end
TetrahedralLimits(::Tuple{}) = error("0-d simplex not possible")
function TetrahedralLimits(a::NTuple{d}) where {d}
    return TetrahedralLimits{d,float(eltype(a))}(ntuple(n -> float(a[n]), Val{d}()), one(float(eltype(a))))
end
TetrahedralLimits(a::Tuple) = TetrahedralLimits(promote(a...))
TetrahedralLimits(a::AbstractVector) = TetrahedralLimits(Tuple(a))

function fixandeliminate(t::TetrahedralLimits{d,T}, x, ::Val{d}) where {d,T}
    return TetrahedralLimits{d-1,T}(Base.front(t.a), convert(T, x)/t.a[d])
end
function segments(t::TetrahedralLimits{d,T}, dim) where {d,T}
    @assert d == dim
    return (zero(T), t.a[d]*t.s)
end


"""
    ProductLimits(lims::AbstractIteratedLimits...)

Construct a collection of limits which yields the first limit followed by the
second, and so on. The inner limits are not allowed to depend on the outer ones.
The outermost variable of integration should be placed first, i.e.
``\\int_{\\Omega} \\int_{\\Gamma}`` should be `ProductLimits(Ω, Γ)`.
Although changing the order of the limits should not change the results, putting the
shortest limits first may save `nested_quadgk` some work.
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
    @assert d==dim
    lim = l.lims[1]
    return segments(lim, ndims(lim))
end



"""
    TranslatedLimits(lims::AbstractIteratedLimits{d}, t::NTuple{d}) where d

Returns the limits of `lims` translated by offsets in `t`.
"""
struct TranslatedLimits{d,C,L,T} <: AbstractIteratedLimits{d,C}
    l::L
    t::T
    TranslatedLimits{d,C}(l::L, t::T) where {d,C,L<:AbstractIteratedLimits{d,C},T<:NTuple{d,C}} =
        new{d,C,L,T}(l, t)
end
function TranslatedLimits(l::AbstractIteratedLimits{d,C}, t::NTuple{d}) where {d,C}
    return TranslatedLimits{d,C}(l, map(x -> convert(C, x), t))
end

function fixandeliminate(t::TranslatedLimits{d,C}, x, ::Val{dim}) where {d,C,dim}
    l = fixandeliminate(t.l, convert(C, x) - t.t[ndims(t)], Val(dim))
    return TranslatedLimits{d-1,C}(l, Base.front(t.t))
end
function segments(t::TranslatedLimits, dim)
    return map(x -> convert(eltype(t), x) + t.t[dim], segments(t.l, dim))
end

# More ideas for limits
# ParametrizedLimits
# requiring linear programming
# RotatedLimits
# AffineLimits

"""
    load_limits(obj)

Load integration limits from an object. Serves as an api hook for package
extensions with specialized limit types.
"""
function load_limits end
