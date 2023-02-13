using Test

using StaticArrays

using IteratedIntegration

@testset "IteratedIntegration" begin
    
    @testset "AbstractLimits" begin

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
            
        end

        @testset "AssociativeOpIntegrand" begin
            # test single variable
            f1 = AssociativeOpIntegrand(+, cos, sin)
            int, = iterated_integration(f1, 0, 2)
            @test int ≈ iterated_integration(sin∘only, 0, 2)[1] + iterated_integration(cos∘only, 0, 2)[1]
            # test multi variable
            # f2 = AssociativeOpIntegrand(+, cos, ThunkIntegrand{1}(sin))
            # int, = iterated_integration(f1, 0, 2)

        end
        #=
        @testset "IteratedIntegrand" begin
            
        end
        =#
    end

    @testset "iterated_integration" begin
        n = 5
        @test iterated_integration(x -> x .^ (1:5), zeros(SVector{n}), ones(SVector{n}))[1] ≈ [1/(i+1) for i in 1:n]
    end

    #=
    @testset "IteratedIntegrator" begin
        
    end
    =#
end