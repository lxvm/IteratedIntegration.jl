"""
A package for iterated integration (IAI). Its main exports are the
[`AbstractIteratedLimits`](@ref) abstraction to evaluate parametrizations of limits of integration,
[`segments`](@ref), a routine that returns the extrema of the domain in a given variable,
[`fixandeliminate`](@ref), to fix a variable of integration and return limits of a reduced dimensionality,
[`eliminate`](@ref), to eliminate or project out a variable of integration,
and [`measure`](@ref) to compute the (hyper)volume of an integration domain.
These functions and the exported domain types are sufficient for implementing iterated integration.
Currently this package does not provide an iterated integration implementation.
"""
module IteratedIntegration

using LinearAlgebra

using StaticArrays
using QuadGK: quadgk

public AbstractIteratedLimits, segments, fixandeliminate, eliminate, load_limits
include("definitions.jl")

public CubicLimits, TetrahedralLimits, ProductLimits, TranslatedLimits
include("iterated_limits.jl")

public measure
include("measure.jl")

end
