"""
A package for iterated adaptive integration (IAI) based on
[`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl).
Its main exports are [`iterated_integration`](@ref), a routine which performs
multidimensional adaptive integration with nested `quadgk` calls,
[`IteratedIntegrator`](@ref), which provides a functor interface to evaluate
integrals, and the [`AbstractLimits`](@ref) abstraction to evaluate
parametrizations of limits of integration.
"""
module IteratedIntegration

using LinearAlgebra

using StaticArrays

using QuadGK: quadgk, do_quadgk, alloc_segbuf
using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays

import IntervalSets: endpoints
import Polyhedra: fixandeliminate

export AbstractIteratedLimits, AbstractIteratedIntegrand
export iterated_pre_eval, iterated_integrand
include("definitions.jl")

export CubicLimits, TetrahedralLimits, PolyhedralLimits, ProductLimits, TranslatedLimits
include("iterated_limits.jl")

export ThunkIntegrand, IteratedIntegrand
include("iterated_integrands.jl")

export iterated_integration # the main routine
include("iterated_integration.jl")

end