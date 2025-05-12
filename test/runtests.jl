using Test

using Polyhedra: polyhedron, vrep, volume
using IteratedIntegration
using IteratedIntegration: fixandeliminate, segments, eliminate, measure, load_limits,
    CubicLimits, TetrahedralLimits, ProductLimits

using Aqua

Aqua.test_all(IteratedIntegration)

@testset "IteratedIntegration" begin
    # TODO: check that segments & fixandeliminate work, and that volume integrates correctly
    @testset "AbstractIteratedLimits" begin

        @testset "CubicLimits" begin
            l3 = CubicLimits(-ones(3), ones(3))
            @test segments(l3, 3) == (-1, 1)
            @test segments(l3, 2) == (-1, 1)
            @test segments(l3, 1) == (-1, 1)
            @test eliminate(l3, 1) == CubicLimits(-ones(2), ones(2))
            @test eliminate(l3, 1:2) == CubicLimits(-ones(1), ones(1))
            @test segments(fixandeliminate(l3,-1, Val(3)), 2) == (-1, 1)
            @test segments(fixandeliminate(l3, 1, Val(3)), 2) == (-1, 1)
            l2 = fixandeliminate(l3, 0.0, Val(3))
            @test segments(l2, 2) == (-1, 1)
            @test segments(l2, 1) == (-1, 1)
            @test eliminate(l2, 2) == CubicLimits(-ones(1), ones(1))
            @test segments(fixandeliminate(l2,-1, Val(2)), 1) == (-1, 1)
            @test segments(fixandeliminate(l2, 1, Val(2)), 1) == (-1, 1)
            l1 = fixandeliminate(l2, 0.1, Val(2))
            @test segments(l1, 1) == (-1, 1)
        end

        @testset "TetrahedralLimits" begin
            t3 = TetrahedralLimits((1,1,1))
            @test segments(t3, 3) == (0, 1)
            @test segments(t3, 2) == (0, 1)
            @test segments(t3, 1) == (0, 1)
            @test segments(fixandeliminate(t3, 0, Val(3)), 2) == (0, 0)
            @test segments(fixandeliminate(t3, 0.5, Val(3)), 2) == (0, 0.5)
            t2 = fixandeliminate(t3, 1, Val(3))
            @test segments(t2, 2) == (0, 1)
            @test segments(t2, 1) == (0, 1)
            @test segments(fixandeliminate(t2, 0, Val(2)), 1) == (0, 0)
            @test segments(fixandeliminate(t2, 0.5, Val(2)), 1) == (0, 0.5)
            t1 = fixandeliminate(t2, 1, Val(2))
            @test segments(t1, 1) == (0, 1)
        end

        @testset "ProductLimits" begin
            l1 = CubicLimits(-1, 1)
            l2 = CubicLimits(fill(-2, 2), fill(2, 2))
            l3 = TetrahedralLimits((3, 3, 3))
            p = ProductLimits(l1, l2, l3)
            @test segments(p, 6) == segments(l1, 1)
            p1 = fixandeliminate(p, 0.0, Val(6))
            @test segments(p1, 5) == segments(l2, 1)
            p2 = fixandeliminate(p1, 0.0, Val(5))
            @test segments(p2, 4) == segments(fixandeliminate(l2, 0.0, Val(2)), 1)
            p3 = fixandeliminate(p2, 0.0, Val(4))
            @test segments(p3, 3) == segments(l3, 3)
            @test segments(fixandeliminate(p3, 0, Val(3)), 2) == segments(fixandeliminate(l3, 0, Val(3)), 2)
            @test segments(fixandeliminate(p3, 1, Val(3)), 2) == segments(fixandeliminate(l3, 1, Val(3)), 2)
            @test segments(fixandeliminate(p3, 2, Val(3)), 2) == segments(fixandeliminate(l3, 2, Val(3)), 2)
            @test segments(fixandeliminate(p3, 3, Val(3)), 2) == segments(fixandeliminate(l3, 3, Val(3)), 2)
        end

        @testset "TranslatedLimits" begin

        end

    end
end

@testset "IteratedIntegrationPolyhedraExt" begin

    p = polyhedron(vrep([
    -1.9 -1.7
    -1.8  0.5
     1.7  0.7
     1.9 -0.3
     0.9 -1.1
    ]))
    pvol = volume(p)

    l = load_limits(p)
    ivol, = measure(l)
    @test pvol ≈ ivol
end
