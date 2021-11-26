using Test
using TestSetExtensions
using MetaBalls

@testset ExtendedTestSet "MetaBalls Tests" begin
    @includetests ["generators_tests", "edge_lookup_tests"]
end