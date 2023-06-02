using Test

using Polyhedra
using IteratedIntegration
using QuadGK
using HCubature

@testset "IteratedIntegration" begin
    # TODO: check that endpoints & fixandeliminate work, and that volume integrates correctly
    @testset "AbstractIteratedLimits" begin

        @testset "CubicLimits" begin

        end

        @testset "TetrahedralLimits" begin

        end

        @testset "ProductLimits" begin

        end

        @testset "TranslatedLimits" begin

        end

    end

    @testset "AbstractIteratedIntegrand" begin

        @testset "ThunkIntegrand" begin
            n = 5
            f = ThunkIntegrand{n}(sum)
            @test f(1:n) == +((1:n)...)
        end

        @testset "IteratedIntegrand" begin
            n = 5
            f = IteratedIntegrand(sum, ntuple(_->identity, n-1)...)
            @test f(1:n) == +((1:n)...)
        end

        @testset "ProductIteratedIntegrand" begin
            n = 5
            f = ProductIteratedIntegrand(sum∘vcat, ThunkIntegrand{n-1}(identity),ThunkIntegrand{1}(identity))
            @test f(1:n) == +((1:n)...)
        end
    end

    @testset "iai" begin
        n = 1
        f = x -> x .^ (1:n)
        a = zeros(n)
        b = ones(n)
        @test iai(f, a, b)[1] ≈ [1/(i+1) for i in 1:n]

        f = IteratedIntegrand(f, ntuple(_->identity, n-1)...)
        @test iai(f, a, b)[1] ≈ [1/(i+1) for i in 1:n]
    end

    @testset "quadgk validation" begin
        @test quadgk(x -> cos(13x), 0,0.7)[1] ≈ iai(x -> cos(13*only(x)), 0,0.7)[1]

        # multiple breakpoints # @test quadgk(cos, 0,0.7,1)[1] ≈ iai(cos∘only, 0,0.7,1)[1]
        # multiple breakpoints # @test quadgk(x -> exp(im*x), 0,0.7,1)[1] ≈ iai(x -> exp(im*only(x)), 0,0.7,1)[1]
        # complex limits # @test quadgk(x -> exp(im*x), 0,1im)[1] ≈ iai(x -> exp(im*only(x)), 0,1im)[1]
        @test isapprox(quadgk(cos, 0,BigFloat(1),order=40)[1], iai(cos∘only, 0,BigFloat(1),order=40)[1],
                       atol=1000*eps(BigFloat))
        # inf limits # @test quadgk(x -> exp(-x), 0,0.7,Inf)[1] ≈ iai(x -> exp(-x), 0,0.7,Inf)[1]
        # inf limits # @test quadgk(x -> exp(x), -Inf,0)[1] ≈ iai(x -> exp(x), -Inf,0)[1]
        # inf limits # @test quadgk(x -> exp(-x^2), -Inf,Inf)[1] ≈ iai(x -> exp(-x^2), -Inf,Inf)[1]
        # inf limits # @test quadgk(x -> [exp(-x), exp(-2x)], 0, Inf)[1] ≈ iai(x -> [exp(-x), exp(-2x)], 0, Inf)[1]
        # multiple breakpoints # @test quadgk(cos, 0,0.7,1, norm=abs)[1] ≈ iai(cos∘only, 0,0.7,1, norm=abs)[1]

        # Test a function that is only implemented for Float32 values
        cos32(x::Float32) = cos(20x)
        @test quadgk(cos32, 0f0, 1f0)[1]::Float32 ≈ iai(cos32∘only, 0f0, 1f0)[1]::Float32

        # test integration of a type-unstable function where the instability is only detected
        # during refinement of the integration interval:
        # @test quadgk(x -> x > 0.01 ? sin(10(x-0.01)) : 1im, 0,1.01, rtol=1e-4, order=3)[1] ≈ (1 - cos(10))/10+0.01im rtol=1e-4

        # order=1 (issue #66)
        @test quadgk_count(x -> 1, 0, 1, order=1) == iai_count(x -> 1, 0, 1, order=1)
    end

    @testset "hcubature validation" begin
        atol = 1e-8
        for n in 1:3, routine in (nested_quadgk, nested_auxquadgk, iai)
            a = zeros(n)
            b = rand(n)
            for integrand in (x -> sin(sum(x)), x -> inv(0.01im+sin(sum(x))))
                @test hcubature(integrand, a, b; atol=atol)[1] ≈ routine(integrand, a, b; atol=atol)[1] atol=2atol
            end
            # test a localized BZ-like integrand
            b = ones(n)*2pi
            integrand = x -> inv(im*10.0^(n-3) + sum(sin, x))
            @test hcubature(integrand, a, b; atol=atol)[1] ≈ routine(integrand, a, b; atol=atol)[1] atol=atol
        end
    end

    @testset "auxquad" begin
        f(x)    = sin(x)/(cos(x)+im*1e-5)   # peaked "nice" integrand
        imf(x)  = imag(f(x))                # peaked difficult integrand
        f2(x)   = f(x)^2                    # even more peaked
        imf2(x) = imf(x)^2                  # even more peaked!

        x0 = 0.1    # arbitrary offset of between peak

        function integrand(x)
            re, im = reim(f2(x) + f2(x-x0))
            AuxValue(imf2(x) + imf2(x-x0), re)
        end

        absI, = auxquadgk(integrand, 0, 2pi, atol=1e-4, parallel=Sequential()) # 628318.5306881254
        relI, = auxquadgk(integrand, 0, 2pi, rtol=1e-6, parallel=Parallel(Float64,AuxValue{Float64},AuxValue{Float64})) # 628318.5306867635
        @test absI.val ≈ relI.val rtol=1e-6
    end

    @testset "inference" begin
        for n in 1:4, routine in (nested_quadgk, nested_auxquadgk) # iai
            a = zeros(n)
            b = rand(n)
            for (integrand,type) in (
                (x -> sin(sum(x)), Tuple{Float64, Float64}),
                (x -> inv(complex(sin(sum(x)), 0.01)), Tuple{ComplexF64,Float64}),
                )
                @inferred type routine(integrand, a, b)
            end
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
    ivol, = nested_quadgk(ThunkIntegrand{2}(x -> 1.0), l)
    @test pvol ≈ ivol
end
