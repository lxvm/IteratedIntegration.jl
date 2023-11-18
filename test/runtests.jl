using Test

using Polyhedra: polyhedron, vrep, volume
using IteratedIntegration
using IteratedIntegration: fixandeliminate, segments
using HCubature
using QuadGK

using Aqua

Aqua.test_all(IteratedIntegration)

@testset "IteratedIntegration" begin
    # TODO: check that segments & fixandeliminate work, and that volume integrates correctly
    @testset "AbstractIteratedLimits" begin

        @testset "CubicLimits" begin
            l3 = CubicLimits(-ones(3), ones(3))
            @test segments(l3, 3) == (-1, 1)
            @test segments(fixandeliminate(l3,-1, Val(3)), 2) == (-1, 1)
            @test segments(fixandeliminate(l3, 1, Val(3)), 2) == (-1, 1)
            l2 = fixandeliminate(l3, 0.0, Val(3))
            @test segments(l2, 2) == (-1, 1)
            @test segments(fixandeliminate(l2,-1, Val(2)), 1) == (-1, 1)
            @test segments(fixandeliminate(l2, 1, Val(2)), 1) == (-1, 1)
            l1 = fixandeliminate(l2, 0.1, Val(2))
            @test segments(l1, 1) == (-1, 1)
        end

        @testset "TetrahedralLimits" begin
            t3 = TetrahedralLimits((1,1,1))
            @test segments(t3, 3) == (0, 1)
            @test segments(fixandeliminate(t3, 0, Val(3)), 2) == (0, 0)
            @test segments(fixandeliminate(t3, 0.5, Val(3)), 2) == (0, 0.5)
            t2 = fixandeliminate(t3, 1, Val(3))
            @test segments(t2, 2) == (0, 1)
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
#=
    @testset "iai" begin
        n = 1
        f = x -> x .^ (1:n)
        a = zeros(n)
        b = ones(n)
        @test iai(f, a, b)[1] ≈ [1/(i+1) for i in 1:n]

        f = IteratedIntegrand(f, ntuple(_->identity, n-1)...)
        @test iai(f, a, b)[1] ≈ [1/(i+1) for i in 1:n]
    end
=#
    @testset "quadgk validation" begin
        @test quadgk(x -> cos(13x), 0,0.7)[1] ≈ auxquadgk(x -> cos(13*only(x)), 0,0.7)[1]

        @test quadgk(cos, 0,0.7,1)[1] ≈ auxquadgk(cos∘only, 0,0.7,1)[1]
        @test quadgk(x -> exp(im*x), 0,0.7,1)[1] ≈ auxquadgk(x -> exp(im*only(x)), 0,0.7,1)[1]
        @test quadgk(x -> exp(im*x), 0,1im)[1] ≈ auxquadgk(x -> exp(im*only(x)), 0,1im)[1]
        @test isapprox(quadgk(cos, 0,BigFloat(1),order=40)[1], auxquadgk(cos∘only, 0,BigFloat(1),order=40)[1],
                       atol=1000*eps(BigFloat))
        @test quadgk(x -> exp(-x), 0,0.7,Inf)[1] ≈ auxquadgk(x -> exp(-x), 0,0.7,Inf)[1]
        @test quadgk(x -> exp(x), -Inf,0)[1] ≈ auxquadgk(x -> exp(x), -Inf,0)[1]
        @test quadgk(x -> exp(-x^2), -Inf,Inf)[1] ≈ auxquadgk(x -> exp(-x^2), -Inf,Inf)[1]
        @test quadgk(x -> [exp(-x), exp(-2x)], 0, Inf)[1] ≈ auxquadgk(x -> [exp(-x), exp(-2x)], 0, Inf)[1]
        @test quadgk(cos, 0,0.7,1, norm=abs)[1] ≈ auxquadgk(cos∘only, 0,0.7,1, norm=abs)[1]

        # Test a function that is only implemented for Float32 values
        cos32(x::Float32) = cos(20x)
        @test quadgk(cos32, 0f0, 1f0)[1]::Float32 ≈ auxquadgk(cos32∘only, 0f0, 1f0)[1]::Float32

        # test integration of a type-unstable function where the instability is only detected
        # during refinement of the integration interval:
        @test auxquadgk(x -> x > 0.01 ? sin(10(x-0.01)) : 1im, 0,1.01, rtol=1e-4, order=3)[1] ≈ (1 - cos(10))/10+0.01im rtol=1e-4

        # order=1 (issue #66)
        @test quadgk_count(x -> 1, 0, 1, order=1) == auxquadgk_count(x -> 1, 0, 1, order=1)
    end

    @testset "hcubature validation" begin
        atol = 1e-8
        for n in 1:3, routine in (nested_quad,)
            a = zeros(n)
            b = ones(n)
            for integrand in (x -> sin(sum(x)), x -> inv(0.01im+sin(sum(x))))
                @test hcubature(integrand, a, b; atol=atol)[1] ≈ routine(integrand, a, b; atol=atol)[1] atol=2atol
            end
            # test a localized BZ-like integrand
            b = ones(n)*2pi
            integrand = x -> inv(im*10.0^(n-3) + sum(sin, x))
            @test hcubature(integrand, a, b; atol=atol)[1] ≈ routine(integrand, a, b; atol=atol)[1] atol=atol
        end
    end

    @testset "auxquadgk" begin
        f(x)    = sin(x)/(cos(x)+im*1e-5)   # peaked "nice" integrand
        imf(x)  = imag(f(x))                # peaked difficult integrand
        f2(x)   = f(x)^2                    # even more peaked
        imf2(x) = imf(x)^2                  # even more peaked!

        x0 = 0.1    # arbitrary offset of between peak

        function integrand(x)
            re, im = reim(f2(x) + f2(x-x0))
            AuxValue(imf2(x) + imf2(x-x0), re)
        end

        absI, = auxquadgk(integrand, 0, 2pi, atol=1e-4) # 628318.5306881254
        relI, = auxquadgk(integrand, 0, 2pi, rtol=1e-6) # 628318.5306867635
        @test absI.val ≈ relI.val rtol=1e-6

        # test the BatchIntegrand interface
        h = IteratedIntegration.AuxQuadGK.BatchIntegrand((y,x) -> y .= integrand.(x), AuxValue{Float64})
        babsI, = auxquadgk(h, 0, pi, 2pi, atol=1e-4)
        @test absI.val ≈ babsI.val atol=1e-4
        brelI, = auxquadgk(h, 0, pi, 2pi, rtol=1e-6)
        @test relI.val ≈ brelI.val rtol=1e-6
    end

    @testset "meroquadgk" begin
        f(x) = inv(complex(0.5, 1e-5) - cos(x-0.1))
        ref, = quadgk(f, 0, 2pi, atol=1e-6)
        val, = meroquadgk(f, 0, 2pi, atol=1e-6)
        @test ref ≈ val atol=1e-6
    end

    @testset "contquadgk" begin
        ff(a) = (q = sqrt(Complex(a^2 - 1)); (abs(q-a) <= 1 ? 1 : -1) * inv(q))
        f(x) = ff(complex(0.5, 1e-5) - cos(x-0.1))
        ref, = quadgk(f, 0, 2pi, atol=1e-6)
        val, = contquadgk(f, 0, 2pi, atol=1e-6)
        @test ref ≈ val atol=1e-6
    end
#=
    @testset "inference" begin
        for n in 1:4, routine in (nested_quad, )
            a = zeros(n)
            b = rand(n)
            for integrand in (
                x -> sin(sum(x)),
                x -> inv(complex(sin(sum(x)), 0.01)),
                )
                @inferred routine(integrand, a, b)
            end
        end
    end
=#
    @testset "nested_quad" begin
        for dims in 2:4
            l = TetrahedralLimits(ntuple(n -> 1.0, dims))
            vol, = nested_quad(x -> 1.0, l)
            @test vol ≈ 1/factorial(dims)
        end
    end
end

@testset "PolyhedraExt" begin

    p = polyhedron(vrep([
    -1.9 -1.7
    -1.8  0.5
     1.7  0.7
     1.9 -0.3
     0.9 -1.1
    ]))
    pvol = volume(p)

    l = load_limits(p)
    ivol, = @inferred(nested_quad(x -> 1.0, l))
    @test pvol ≈ ivol
end
