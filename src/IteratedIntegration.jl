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
using FunctionWrappers: FunctionWrapper
using QuadGK: alloc_segbuf, quadgk, quadgk_count, quadgk_print

export AbstractIteratedLimits
include("definitions.jl")

export CubicLimits, TetrahedralLimits, ProductLimits, TranslatedLimits, load_limits
include("iterated_limits.jl")

export AuxValue
include("AuxQuadGK/AuxQuadGK.jl")
using .AuxQuadGK

include("MeroQuadGK/MeroQuadGK.jl")
using .MeroQuadGK

include("ContQuadGK/ContQuadGK.jl")
using .ContQuadGK

include("nested_quad.jl")

export quadgk, quadgk_count, quadgk_print
for routine in (:auxquadgk, :auxquadgk!, :contquadgk, :meroquadgk, :nested_quad)
    routine_count = Symbol(routine, :_count)
    routine_print = Symbol(routine, :_print)

    @eval begin
        export $routine, $routine_count, $routine_print

        function $routine_count(f, args...; kwargs...)
            numevals::Int = 0
            I, E = $routine(args...; kwargs...) do x
                numevals += 1
                return f(x)
            end
            return (I, E, numevals)
        end

        $routine_print(io::IO, f, args...; kws...) = $routine_count(args...; kws...) do x
            y = f(x)
            println(io, "f(", x, ") = ", y)
            y
        end
        $routine_print(f, args...; kws...) = $routine_print(stdout, f, args...; kws...)
    end
end

end
