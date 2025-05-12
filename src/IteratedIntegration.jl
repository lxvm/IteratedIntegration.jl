"""
A package for iterated adaptive integration (IAI) based on
[`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl). Its main exports are
[`nested_quad`](@ref), a routine which performs multidimensional adaptive
integration with nested `quadgk` calls and the [`AbstractIteratedLimits`](@ref)
abstraction to evaluate parametrizations of limits of integration.
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
