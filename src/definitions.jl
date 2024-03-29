"""
    AbstractIteratedLimits{d,T}

Supertype for limits of integration over a domain with elements of type `SVector{d,T}`.
In order to work with iterated integration, the following methods must be implemented

# Interface
- [`segments`](@ref): returns an iterator over intervals to integrate in the current dimension
- [`fixandeliminate`](@ref): return another limit object with one of the variables of
  integration eliminated

the domain of integration must be convex.
"""
abstract type AbstractIteratedLimits{d,T<:Number} end


Base.ndims(::AbstractIteratedLimits{d}) where {d} = d
Base.eltype(::Type{<:AbstractIteratedLimits{d,T}}) where {d,T} = T

limit_iterate(l::AbstractIteratedLimits) = segments(l, ndims(l)), l, ()
function limit_iterate(l::AbstractIteratedLimits, state, x)
    lx = fixandeliminate(l, x, Val(ndims(l)))
    return segments(lx, ndims(lx)), lx, (x, state...)
end
function limit_iterate(::AbstractIteratedLimits{1}, state, x)
    return SVector(promote(x, state...))
end

# return a point inside the limits of integration that can be evaluated
function interior_point(l::AbstractIteratedLimits)
    a, b, = segments(l, ndims(l))
    x = (a + b)/2
    ndims(l) === 1 && return SVector{1,typeof(x)}(x)
    lx = fixandeliminate(l, x, Val(ndims(l)))
    return SVector{ndims(l),typeof(x)}((interior_point(lx)...,x))
end


"""
    segments(::AbstractLimits, dim)

Return an iterator over endpoints and breakpoints in the limits along dimension `dim`.
They must be sorted.
"""
function segments end

"""
    fixandeliminate(l::AbstractIteratedLimits, x)

Fix the outermost variable of integration and return the inner limits.

!!! note "For developers"
    Realizations of type `T<:AbstractIteratedLimits` only have to implement a method
    with signature `fixandeliminate(::T, ::Number)`. The result must also have
    dimension one less than the input, and this should only be called when ndims
    >= 1
"""
function fixandeliminate end

# we pass in segs that may be of variable length, but quadgk inference is bad in that case.
# auxquadgk should be able to handle both efficiently
myquadgk(f, segs::NTuple; kws...) = quadgk(f, segs...; kws...)
myquadgk(f, segs::AbstractVector; kws...) = quadgk(f, segs...; kws...)
myquadgk(args...; kws...) = quadgk(args...; kws...)
