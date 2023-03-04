"""
A package for iterated adaptive integration (IAI) based on
[`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl). Its main exports are
[`nested_quadgk`](@ref), a routine which performs multidimensional adaptive
integration with nested `quadgk` calls, [`iai`](@ref), which performs
globally-adaptive iterated integration, and the [`AbstractIteratedLimits`](@ref)
abstraction to evaluate parametrizations of limits of integration.
"""
module IteratedIntegration

using LinearAlgebra

using StaticArrays

using AbstractTrees
using DataStructures: BinaryMaxHeap, extract_all!
using QuadGK: quadgk, do_quadgk, alloc_segbuf, cachedrule, Segment
using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays

import Base.Order.Reverse
import QuadGK: evalrule
import IntervalSets: endpoints
import Polyhedra: fixandeliminate

export AbstractIteratedLimits, AbstractIteratedIntegrand
export iterated_pre_eval, iterated_integrand
include("definitions.jl")

export CubicLimits, TetrahedralLimits, PolyhedralLimits, ProductLimits, TranslatedLimits
include("iterated_limits.jl")

export ThunkIntegrand, IteratedIntegrand
include("iterated_integrands.jl")

export nested_quadgk
include("nested_quadgk.jl")

export iai, iai_buffer, iai_count, iai_print
include("iai.jl")

end