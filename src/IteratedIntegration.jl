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
using FunctionWrappers: FunctionWrapper
using QuadGK: alloc_segbuf


export AbstractIteratedLimits
include("definitions.jl")

export CubicLimits, TetrahedralLimits, ProductLimits, TranslatedLimits, load_limits
include("iterated_limits.jl")


export auxquadgk, AuxValue, Sequential, Parallel
include("RuleQuad.jl")
using .RuleQuad
import .RuleQuad: countevals

export nested_quad
include("nested_quad.jl")

for routine in (:nested_quad, :auxquadgk)
    routine_count = Symbol(routine, :_count)
    routine_print = Symbol(routine, :_print)

    @eval export $routine_count, $routine_print

    @eval function $routine_count(f, args...; kwargs...)
        numevals = Threads.Atomic{Int}(0)
        I, E = $routine(args...; kwargs...) do x
            Threads.atomic_add!(numevals, 1)
            return f(x)
        end
        return (I, E, numevals[])
    end

    @eval $routine_print(io::IO, f, args...; kws...) = $routine_count(args...; kws...) do x
        y = f(x)
        println(io, "f(", x, ") = ", y)
        y
    end
    @eval $routine_print(f, args...; kws...) = $routine_print(stdout, f, args...; kws...)
end

end
