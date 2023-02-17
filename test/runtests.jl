using Test

using IteratedIntegration

@testset "IteratedIntegration" begin
    # TODO: check that endpoints & fixandeliminate work, and that volume integrates correctly
    @testset "AbstractIteratedLimits" begin

        @testset "CubicLimits" begin
            
        end

        @testset "TetrahedralLimits" begin
            
        end

        @testset "PolyhedralLimits" begin
            
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
    end

    @testset "iterated_integration" begin
        n = 5
        f = x -> x .^ (1:n)
        a = zeros(n)
        b = ones(n)
        @test iterated_integration(f, a, b)[1] ≈ [1/(i+1) for i in 1:n]

        f = IteratedIntegrand(f, ntuple(_->identity, n-1)...)
        @test iterated_integration(f, a, b)[1] ≈ [1/(i+1) for i in 1:n]
    end

end