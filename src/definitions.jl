"""
    AbstractIteratedLimits{d,T}

Supertype for limits of integration over a domain with elements of type `SVector{d,T}`.
In order to work with iterated integration, the following methods must be implemented

# Interface
- [`nextdim`](@ref): returns `Val(n)` where `n` is the next variable of integration
- [`segments`](@ref): returns an iterator over intervals to integrate in the current dimension
- [`fixandeliminate`](@ref): return another limit object with one of the variables of
  integration eliminated
"""
abstract type AbstractIteratedLimits{d,T<:Number} end

Base.ndims(::AbstractIteratedLimits{d}) where {d} = d
Base.eltype(::Type{<:AbstractIteratedLimits{d,T}}) where {d,T} = T


"""
    segments(::AbstractLimits)

Return a `segitr`, i.e. an iterator over interval
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
