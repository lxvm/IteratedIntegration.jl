"""
    AbstractIteratedLimits{d,T}

Supertype for limits that work with iterated integration
"""
abstract type AbstractIteratedLimits{d,T<:AbstractFloat} end

Base.ndims(::AbstractIteratedLimits{d}) where {d} = d
Base.eltype(::Type{<:AbstractIteratedLimits{d,T}}) where {d,T} = T


"""
    endpoints(::AbstractLimits)

Return a tuple with the extrema of the next variable of integration. This is
equivalent to projecting the integration domain onto one dimension.
"""
function endpoints end

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


"""
    AbstractIteratedIntegrand{d}

Supertype for integrands compatible with iterated integration of `d` variables.

    (::AbstractIteratedIntegrand)(x)

Evaluate the integrand at the point `x`
"""
abstract type AbstractIteratedIntegrand{d} end

# interface

"""
    iterated_pre_eval(f::AbstractIteratedIntegrand{d}, x, ::Val{d}) where d

Perform a precomputation on `f` using the value of a variable of integration,
`x`. Certain types of functions, such as Fourier series, take can use `x` to
precompute a new integrand for the remaining variables of integration that is
more computationally efficient. Otherwise, the type can store `x` and delay the
evaluation to the inner integral. This function must return the integrand for
the subsequent integral, which should be an `AbstractIteratedIntegrand{d-1}`
"""
function iterated_pre_eval end

"""
    iterated_integrand(f::AbstractIteratedIntegrand{d}, x, ::Val{d}) where d

Evaluate a function on the inner integral `x`. When `d==1`, `x` is the argument
of the innermost integral that is passed by the integration routine via `f(x)`.
"""
function iterated_integrand end

# abstract methods

(f::AbstractIteratedIntegrand{1})(x::NTuple{1}) =
    iterated_integrand(f, x[1], Val(1))
(f::AbstractIteratedIntegrand)(::Tuple{}) = f
(f::AbstractIteratedIntegrand{d})(x::NTuple{N}) where {d,N} =
    iterated_pre_eval(f, x[N], Val(d))(x[1:N-1])
(f::AbstractIteratedIntegrand)(x) = f(promote(x...))
