using Test
using TestSetExtensions
using ReferenceTests
using MetaBalls:MetaBalls

include("generators.jl")
include("c_code.jl")

@test_reference "voxels.txt" map((x) -> edge_lookup(x), generate_voxels(16))

@testset "edge lookup" begin
    @test_reference "voxels.txt" map((x) -> MetaBalls.edge_lookup(x), generate_voxels(16))
end