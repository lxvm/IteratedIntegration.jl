using Test
using IteratedIntegration: load_limits, measure, eliminate
using Polyhedra: polyhedron, vrep, volume
using QuadGK: quadgk

@testset "measure" begin
    p = polyhedron(vrep([
        -1.9 -1.7
        -1.8  0.5
        1.7  0.7
        1.9 -0.3
        0.9 -1.1
    ]))
    pvol = volume(p)

    l = load_limits(p)
    ivol = measure(quadgk, l)
    @test pvol ≈ ivol
end
@testset "eliminate" begin
    p = polyhedron(vrep([
        0.0 0.0
        0.0 0.5
        0.3 0.5
        0.3 0.0
    ]))
    l = load_limits(p)
    l0 = eliminate(l, 2)
    ivol = measure(quadgk, l0)
    @test ivol ≈ 0.3
end