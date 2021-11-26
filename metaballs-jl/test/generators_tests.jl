using Test
using TestSetExtensions
using MetaBalls:MetaBalls

include("generators.jl")

@testset "generate_voxels" begin
    x = generate_voxels(4)

        # printstyled(x[1], "\n"; color=:blue)
        # printstyled(size(x[1], 1), "\n"; color=:blue)
    
    @test size(x[1], 1) == 8
    @test size(x, 1) == 27
end
