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
function iterate(::AbstractIteratedLimits{d}, v::Val{dim}=Val(d)) where {d,dim}
    dim === 1 && return nothing
    return (v, Val(dim-1))
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
