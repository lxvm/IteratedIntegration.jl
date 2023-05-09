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

using DataStructures: BinaryMaxHeap, extract_all!
using QuadGK: quadgk, do_quadgk, alloc_segbuf, cachedrule, Segment
using AuxQuad: do_auxquadgk, Integrands, Errors
using Polyhedra: Polyhedron, VRepresentation, vrep, points, fulldim, hasallrays, coefficient_type

import Base.Order.Reverse
import QuadGK: evalrule
import IntervalSets: endpoints
import Polyhedra: fixandeliminate

export AbstractIteratedLimits, AbstractIteratedIntegrand
export iterated_pre_eval, iterated_integrand
include("definitions.jl")

export CubicLimits, TetrahedralLimits, PolyhedralLimits, ProductLimits, TranslatedLimits
include("iterated_limits.jl")

export ThunkIntegrand, IteratedIntegrand, ProductIteratedIntegrand
include("iterated_integrands.jl")

export nested_quadgk
include("nested_quadgk.jl")

export nested_auxquadgk, Integrands, Errors
include("nested_auxquadgk.jl")

export iai, iai_buffer
include("iai.jl")

for routine in (:nested_quadgk, :iai)
    routine_count = Symbol(routine, :_count)
    routine_print = Symbol(routine, :_print)

    @eval export $routine_count, $routine_print

    @eval $routine_count(f, a, b; kwargs...) =
        $routine_count(f, CubicLimits(a, b); kwargs...)
    @eval function $routine_count(f, l; kwargs...)
        numevals = 0
        g = ThunkIntegrand{ndims(l)}(x -> (numevals += 1; f(x)))
        I, E = $routine(g, l; kwargs...)
        return (I, E, numevals)
    end

    @eval $routine_print(io::IO, f, args...; kws...) = $routine_count(args...; kws...) do x
        y = f(x)
        println(io, "f(", x, ") = ", y)
        y
    end
    @eval $routine_print(f, args...; kws...) = $routine_print(stdout, f, args...; kws...)
end

end
