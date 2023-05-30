"""
    CubicLimits(a, b)

Store integration limit information for a hypercube with vertices `a` and `b`.
which can be can be real numbers, tuples, or `AbstractVector`s.
The outermost variable of integration corresponds to the last entry.
"""
struct CubicLimits{d,T,L} <: AbstractIteratedLimits{d,T}
    a::L
    b::L
    CubicLimits{d,T}(a::L, b::L) where {d,T,L<:NTuple{d,T}} = new{d,T,L}(a, b)
end
function CubicLimits(a::NTuple{d,A}, b::NTuple{d,B}) where {d,A<:Real,B<:Real}
    T = float(promote_type(A, B))
    CubicLimits{d,T}(
        ntuple(n -> T(a[n]), Val{d}()),
        ntuple(n -> T(b[n]), Val{d}()),
    )
end
CubicLimits(a::Real, b::Real) = CubicLimits((a,), (b,))
CubicLimits(a::AbstractVector, b::AbstractVector) = CubicLimits(Tuple(a), Tuple(b))

function endpoints(c::CubicLimits{d}, dim=d) where d
    1 <= dim <= d || throw(ArgumentError("pick dim=$(dim) in 1:$d"))
    return (c.a[dim], c.b[dim])
end

fixandeliminate(c::CubicLimits{d,T}, _) where {d,T} =
    CubicLimits{d-1,T}(Base.front(c.a), Base.front(c.b))

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
TetrahedralLimits(a::NTuple{d,T}) where {d,T} =
    TetrahedralLimits{d,float(T)}(ntuple(n -> float(a[n]), Val{d}()), one(T))
TetrahedralLimits(a::Tuple) = TetrahedralLimits(promote(a...))
TetrahedralLimits(a::AbstractVector) = TetrahedralLimits(Tuple(a))

endpoints(t::TetrahedralLimits{d,T}) where {d,T} = (zero(T), t.a[d]*t.s)

fixandeliminate(t::TetrahedralLimits{d,T}, x) where {d,T} =
    TetrahedralLimits{d-1,T}(Base.front(t.a), convert(T, x)/t.a[d])


function corners(t::AbstractIteratedLimits)
    a, b = endpoints(t)
    ndims(t) == 1 && return [(a,), (b,)]
    ta = corners(fixandeliminate(t, a))
    tb = corners(fixandeliminate(t, b))
    unique((map(x -> (x..., a), ta)..., map(x -> (x..., b), tb)...))
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

endpoints(l::ProductLimits) = endpoints(l.lims[1])

fixandeliminate(l::ProductLimits{d,T}, x) where {d,T} =
    ProductLimits{d-1,T}(Base.setindex(l.lims, fixandeliminate(l.lims[1], x), 1))
fixandeliminate(l::ProductLimits{d,T,<:Tuple{<:AbstractIteratedLimits{1},Vararg{AbstractIteratedLimits}}}, x) where {d,T} =
    ProductLimits{d-1,T}(Base.tail(l.lims))


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
TranslatedLimits(l::AbstractIteratedLimits{d,C}, t::NTuple{d}) where {d,C} =
    TranslatedLimits{d,C}(l, map(x -> convert(C, x), t))

endpoints(t::TranslatedLimits) =
    map(x -> x + t.t[ndims(t)], endpoints(t.l))
fixandeliminate(t::TranslatedLimits{d,C}, x) where {d,C} =
    TranslatedLimits{d-1,C}(fixandeliminate(t.l, convert(C, x) - t.t[ndims(t)]), Base.front(t.t))

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
