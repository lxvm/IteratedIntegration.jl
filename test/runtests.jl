using Test

@testset "aqua" include("aqua.jl")
@testset "iterated_limits" include("iterated_limits.jl")
@testset "measure" include("measure.jl")
@testset "IteratedIntegrationQuadGKExt" include("IteratedIntegrationQuadGKExt.jl")
@testset "IteratedIntegrationPolyhedraExt" include("IteratedIntegrationPolyhedraExt.jl")
