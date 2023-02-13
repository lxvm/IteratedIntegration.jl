"""
A package for iterated adaptive integration (IAI) based on
[`QuadGK.jl`](https://github.com/JuliaMath/QuadGK.jl).
"""
module IteratedIntegration

using LinearAlgebra

using StaticArrays

using QuadGK: quadgk, do_quadgk, alloc_segbuf
using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays

import IntervalSets: endpoints
import Polyhedra: fixandeliminate, coefficient_type

export AbstractLimits, endpoints, fixandeliminate, coefficient_type
export CubicLimits, TetrahedralLimits, PolyhedralLimits, ProductLimits, TranslatedLimits
include("iterated_limits.jl")

export AbstractIteratedIntegrand, nvars, iterated_pre_eval, iterated_integrand
export ThunkIntegrand, AssociativeOpIntegrand#, IteratedIntegrand # in progress
include("iterated_integrands.jl")

export iterated_integration # the main routine
export iterated_tol_update, iterated_segs, iterated_inference, iterated_integral_type
include("iterated_integration.jl")

export AbstractIteratedIntegrand, limits, quad_integrand, quad_routine, quad_args, quad_kwargs
export IteratedIntegrator
include("iterated_integrator.jl")

end