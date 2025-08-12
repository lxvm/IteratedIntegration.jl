using Test
using IteratedIntegration: CubicLimits, TetrahedralLimits, ProductLimits, TranslatedLimits
using IteratedIntegration: segments, eliminate, fixandeliminate, interior_point
using LinearAlgebra

@testset "CubicLimits" begin
    l3 = CubicLimits(-ones(3), ones(3))
    @test segments(l3, 3) == (-1, 1)
    @test segments(l3, 2) == (-1, 1)
    @test segments(l3, 1) == (-1, 1)
    @test eliminate(l3, 1) == CubicLimits(-ones(2), ones(2))
    @test eliminate(l3, 1:2) == CubicLimits(-ones(1), ones(1))
    @test eliminate(l3, 1:3) === nothing
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
    @test all(l3.a .< interior_point(l3) .< l3.b)
end

# https://math.stackexchange.com/questions/1226707/how-to-check-if-point-x-in-mathbbrn-is-in-a-n-simplex
function insimplex(l::TetrahedralLimits, x::AbstractVector)
    @assert ndims(l) == length(x)
    T = zeros(eltype(l), ndims(l), ndims(l))
    T .= l.a
    for i in 1:ndims(l)
        for j in 1:i-1
            T[j,i] = zero(eltype(l))
        end
    end
    v = LowerTriangular(T) \ x
    any(<(0), v) && return false
    all(<=(1), v) && return true
    return false
end

@testset "TetrahedralLimits" begin
    t3 = TetrahedralLimits((1,1,1))
    @test t3 == TetrahedralLimits([1,1,1])
    @test eliminate(t3, 3) == TetrahedralLimits((1,1))
    @test eliminate(t3, 2:3) == TetrahedralLimits((1,))
    @test eliminate(t3, 1:3) === nothing
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
    @test_throws "not possible" TetrahedralLimits(())
    @test insimplex(t3, interior_point(t3))
    @test insimplex(t2, interior_point(t2))
    @test insimplex(t1, interior_point(t1))
end

@testset "ProductLimits" begin
    l1 = CubicLimits(-1, 1)
    l2 = CubicLimits((-1, -2), (2, 1))
    l3 = TetrahedralLimits((1, 2, 3))
    p = ProductLimits(l1, l2, l3)
    @test segments(p, 6) == segments(l1, 1)
    p1 = fixandeliminate(p, 0.0, Val(6))
    @test segments(p, 5) == segments(l2, 2)
    @test segments(p1, 5) == segments(l2, 2)
    p2 = fixandeliminate(p1, 0.0, Val(5))
    @test segments(p2, 4) == segments(fixandeliminate(l2, 0.0, Val(2)), 1)
    p3 = fixandeliminate(p2, 0.0, Val(4))
    @test segments(p3, 3) == segments(l3, 3)
    @test segments(fixandeliminate(p3, 0, Val(3)), 2) == segments(fixandeliminate(l3, 0, Val(3)), 2)
    @test segments(fixandeliminate(p3, 1, Val(3)), 2) == segments(fixandeliminate(l3, 1, Val(3)), 2)
    @test segments(fixandeliminate(p3, 2, Val(3)), 2) == segments(fixandeliminate(l3, 2, Val(3)), 2)
    @test segments(fixandeliminate(p3, 3, Val(3)), 2) == segments(fixandeliminate(l3, 3, Val(3)), 2)
    x = interior_point(p)
    @test all(l1.a .< x[6] .< l1.b) && all(l2.a .< x[4:5] .< l2.b) && insimplex(l3, x[1:3])
    @test eliminate(p, 6) == ProductLimits(l2, l3)
    @test eliminate(p, 4:6) == ProductLimits(l3)
    @test eliminate(p, 2:3) == ProductLimits(l1, l2, eliminate(l3, 2:3))
    @test eliminate(p, 1) == ProductLimits(l1, l2, eliminate(l3, 1))
    @test eliminate(p, 1:3) == ProductLimits(l1, l2)
    @test eliminate(p, [1,4,5,6]) == ProductLimits(eliminate(l3, 1))
    @test eliminate(p, 2:6) == ProductLimits(eliminate(l3, 2:3))
    @test eliminate(p, 1:6) === nothing
    @test eliminate(p, 3) == ProductLimits(l1, l2, eliminate(l3, 3))
    @test eliminate(p, [3,6]) == ProductLimits(l2, eliminate(l3, 3))
    @test eliminate(p, [3,5,6]) == ProductLimits(eliminate(l2, 2), eliminate(l3, 3))
end

@testset "TranslatedLimits" begin
    l3 = TetrahedralLimits((1, 2, 3))
    q = (5.0, -4.0, 3.0)
    t = TranslatedLimits(l3, q)
    @test segments(t, 3) == segments(l3, 3) .+ q[3]
    @test segments(t, 2) == segments(l3, 2) .+ q[2]
    @test segments(t, 1) == segments(l3, 1) .+ q[1]
    @test fixandeliminate(t, 5.0, Val(3)) == TranslatedLimits(fixandeliminate(l3, 5.0-q[3], Val(3)), q[1:2])
    @test eliminate(t, 3) == TranslatedLimits(eliminate(l3, 3), q[1:2])
    @test insimplex(l3, interior_point(t).-q)
end
