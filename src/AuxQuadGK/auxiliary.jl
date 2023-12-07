struct AuxValue{T,A}
    val::T
    aux::A
end

for op in (:zero, :one, :oneunit)
    @eval Base.$op(a::AuxValue) = AuxValue($op(a.val), $op(a.aux))
    @eval Base.$op(::Type{AuxValue{T,A}}) where {T,A} = AuxValue($op(T), $op(A))
end

struct KeyOrdering{T<:Base.Order.Ordering} <: Base.Order.Ordering
    o::T
    k::Symbol
end
Base.Order.lt(o::KeyOrdering, a::Number, b::AuxValue) =
    Base.Order.lt(o.o, a, getproperty(b, o.k))
Base.Order.lt(o::KeyOrdering, a::AuxValue, b::Number) =
    Base.Order.lt(o.o, getproperty(a, o.k), b)
Base.Order.lt(o::KeyOrdering, a::AuxValue, b::AuxValue) =
    Base.Order.lt(o.o, getproperty(a, o.k), getproperty(b, o.k))

const AuxSegment = Segment{<:Any,<:AuxValue,<:AuxValue}

Base.Order.lt(o::KeyOrdering, a::Number, b::AuxSegment) =
    Base.Order.lt(o, a, b.E)
Base.Order.lt(o::KeyOrdering, a::AuxSegment, b::Number) =
    Base.Order.lt(o, a.E, b)
Base.Order.lt(o::KeyOrdering, a::AuxSegment, b::AuxSegment) =
    Base.Order.lt(o, a.E, b.E)

# first refine the auxiliary, then the true value
eachorder(::AuxValue) = (KeyOrdering(Reverse, :aux), KeyOrdering(Reverse, :val))
eachorder(::Any) = (Reverse,)   # fallback to normal quadgk ordering for other types

LinearAlgebra.norm(a::AuxValue) = AuxValue(norm(a.val), norm(a.aux))
# we treat AuxValue like a view into just that value part
Base.size(a::AuxValue) = size(a.val)
Base.eltype(::Type{AuxValue{T,A}}) where {T,A} = eltype(T)

Base.:+(a::AuxValue, b::AuxValue) = AuxValue(a.val+b.val, a.aux+b.aux)
Base.:-(a::AuxValue, b::AuxValue) = AuxValue(a.val-b.val, a.aux-b.aux)
Base.:*(a::AuxValue, b::AuxValue) = AuxValue(a.val*b.val, a.aux*b.aux)
Base.:*(a::AuxValue, b) = AuxValue(a.val*b, a.aux*b)
Base.:*(a, b::AuxValue) = AuxValue(a*b.val, a*b.aux)
Base.:/(a::AuxValue, b) = AuxValue(a.val/b, a.aux/b)
Base.:/(a, b::AuxValue) = AuxValue(a/b.val, a/b.aux)
Base.:/(a::AuxValue, b::AuxValue) = AuxValue(a.val/b.val, a.aux/b.aux)

Base.isinf(a::AuxValue) = isinf(a.val) || isinf(a.aux)
Base.isnan(a::AuxValue) = isnan(a.val) || isnan(a.aux)

Base.isless(a::AuxValue, b::AuxValue) = isless(a.aux, b.aux) && isless(a.val, b.val)
Base.isless(a::AuxValue, b) = isless(a.aux, b) && isless(a.val, b)
Base.isless(a::AuxValue, b::Missing) = isless(a.aux, b) && isless(a.val, b)
Base.isless(a, b::AuxValue) = isless(a, b.aux) && isless(a, b.val)
Base.isless(a::Missing, b::AuxValue) = isless(a, b.aux) && isless(a, b.val)

# strict error comparisons (De Morgan's Laws)
Base.:>(a::AuxValue, b::AuxValue) = >(a.val, b) || >(a.aux, b)
Base.:>(a::AuxValue, b) = >(a.val, b) || >(a.aux, b)
Base.:>(a, b::AuxValue) = >(a, b.val) || >(a, b.aux)

Base.:<(a::AuxValue, b::AuxValue) = <(a.val, b) && <(a.aux, b)
Base.:<(a::AuxValue, b) = <(a.val, b) && <(a.aux, b)
Base.:<(a::AuxValue, b::Missing) = <(a.val, b) && <(a.aux, b)
Base.:<(a, b::AuxValue) = <(a, b.val) && <(a, b.aux)
Base.:<(a::Missing, b::AuxValue) = <(a, b.val) && <(a, b.aux)

Base.isequal(a::AuxValue, b::AuxValue) = isequal(a.val, b.val) && isequal(a.aux, b.aux)
Base.isequal(a::AuxValue, b) = isequal(a.val, b) && isequal(a.aux, b)
Base.isequal(a::AuxValue, b::Missing) = isequal(a.val, b) && isequal(a.aux, b)
Base.isequal(a, b::AuxValue) = isequal(a, b.val) && isequal(a, b.aux)
Base.isequal(a::Missing, ::AuxValue) = isequal(a, b.val) && isequal(a, b.aux)
Base.max(a::AuxValue, b::AuxValue) = AuxValue(max(a.val, b.val), max(a.aux, b.aux))
Base.max(a::AuxValue, b) = AuxValue(max(a.val, b), max(a.aux, b))
Base.max(a::AuxValue, b::Missing) = AuxValue(max(a.val, b), max(a.aux, b))
Base.max(a, b::AuxValue) = AuxValue(max(a, b.val), max(a, b.aux))
Base.max(a::Missing, b::AuxValue) = AuxValue(max(a, b.val), max(a, b.aux))
function Base.isapprox(a::AuxValue, b::AuxValue; kwargs...)
    return isapprox(a.val, b.val; kwargs...) & isapprox(a.aux, b.aux; kwargs...)
end
