using Test
using IteratedIntegration: CubicLimits, TetrahedralLimits, ProductLimits, TranslatedLimits
using IteratedIntegration: measure
using QuadGK
using LinearAlgebra

l = CubicLimits((-1,-1,-1), (1,1,1))
@test measure(quadgk, l) ≈ prod(l.b .- l.a)
l = TetrahedralLimits((1,2,3))
@test measure(quadgk, l) ≈ abs(det([0;0;3;;0;2;3;;1;2;3])) / factorial(3) atol=1e-8
l = ProductLimits(CubicLimits((-1, -1), (1, 1)), CubicLimits((3.0,), (5.0,)))
@test measure(quadgk, l) ≈ prod((1, 1) .- (-1, -1)) * 2
l = TranslatedLimits(TetrahedralLimits((1,2,3)), (-3,-4,-5))
@test measure(quadgk, l) ≈ abs(det([0;0;3;;0;2;3;;1;2;3])) / factorial(3) atol=1e-8