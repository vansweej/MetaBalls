# plotting with GLMakie
# please refer to http://makie.juliaplots.org/stable/plotting_functions/scatter.html

module MetaBalls

using GLMakie

export Vertex, VoxelElement, Voxel

include("lookup.jl")

BALL_POS = [(8.5, 8.5, 8.5), (8.5, 17.0, 8.5)] # ball position in in the middle

struct Vertex
    x::Float64
    y::Float64
    z::Float64
end

struct VoxelElement
    vertex::Vertex
    iso_value::Float64
end

const Voxel = Array{VoxelElement,1}

# size of grid in 3 dimensions
# the zero point of the grid is in the midle of the cube
# so for a grid 16 sized that is (8.5, 8.5, 8.5) (julia is 1 indexed)
GRID_SIZE = 32

ISO_VALUE = 0.1
ISO_RANGE = 0.05

function distance(x, y, z, posArr)
    fields = [1.0 / ((pos[1] - x)^2 + (pos[2] - y)^2 + (pos[3] - z)^2) for pos in posArr]

    return fields
end

function fill_ISO_SURFACE(size)
    iso_surfaces = [distance(x - 1, y - 1, z - 1, BALL_POS) for x = 1:size, y = 1:size, z = 1:size]
    result = map((x) -> sum(x), iso_surfaces)

    return result
end

function edge_lookup(voxel::Voxel)
    cubeIndex = 0

    for (index, voxelElement) in enumerate(voxel)
        if (voxelElement.iso_value < ISO_VALUE)
            cubeIndex |= 1 << (index - 1)
        end
    end

    return cubeIndex
end

# p1 and p2 are actually points of the cubic grid in local/world coordinates
# here we simply uses the indexes of the isosurfaces table
# valp are the iso values at every vertex
function edgeInterpolate(isolevel, p1, p2, valp1, valp2)
    # println(isolevel, "  ", p1, "  ", p2, "  ", valp1, "  ", valp2)

    if (abs(isolevel - valp1) < 0.00001)
        return (p1)
    end
    if (abs(isolevel - valp2) < 0.00001)
        return (p2)
    end
    if (abs(valp1 - valp2) < 0.00001)
        return (p1)
    end

    mu = (isolevel - valp1) / (valp2 - valp1)

    x = p1.x + mu * (p2.x - p1.x)
    y = p1.y + mu * (p2.y - p1.y)
    z = p1.z + mu * (p2.z - p1.z)

    return Vertex(x, y, z)
end

function vertices(edges, voxel, isolevel)
    vertList = Array{Union{Missing,Vertex}}(missing, 12)

    if (edges & 1 == 1)
        vertList[1] = edgeInterpolate(isolevel, voxel[1].vertex, voxel[2].vertex, voxel[1].iso_value, voxel[2].iso_value)
    end
    if (edges & 2 == 2)
        vertList[2] = edgeInterpolate(isolevel, voxel[2].vertex, voxel[3].vertex, voxel[2].iso_value, voxel[3].iso_value)
    end
    if (edges & 4 == 4)
        vertList[3] = edgeInterpolate(isolevel, voxel[3].vertex, voxel[4].vertex, voxel[3].iso_value, voxel[4].iso_value)
    end
    if (edges & 8 == 8)
        vertList[4] = edgeInterpolate(isolevel, voxel[4].vertex, voxel[1].vertex, voxel[4].iso_value, voxel[1].iso_value)
    end
    if (edges & 16 == 16)
        vertList[5] = edgeInterpolate(isolevel, voxel[5].vertex, voxel[6].vertex, voxel[5].iso_value, voxel[6].iso_value)
    end
    if (edges & 32 == 32)
        vertList[6] = edgeInterpolate(isolevel, voxel[6].vertex, voxel[7].vertex, voxel[6].iso_value, voxel[7].iso_value)
    end
    if (edges & 64 == 64)
        vertList[7] = edgeInterpolate(isolevel, voxel[7].vertex, voxel[8].vertex, voxel[7].iso_value, voxel[8].iso_value)
    end
    if (edges & 128 == 128)
        vertList[8] = edgeInterpolate(isolevel, voxel[8].vertex, voxel[5].vertex, voxel[8].iso_value, voxel[5].iso_value)
    end
    if (edges & 256 == 256)
        vertList[9] = edgeInterpolate(isolevel, voxel[1].vertex, voxel[5].vertex, voxel[1].iso_value, voxel[5].iso_value)
    end
    if (edges & 512 == 512)
        vertList[10] = edgeInterpolate(isolevel, voxel[2].vertex, voxel[6].vertex, voxel[2].iso_value, voxel[6].iso_value)
    end
    if (edges & 1024 == 1024)
        vertList[11] = edgeInterpolate(isolevel, voxel[3].vertex, voxel[7].vertex, voxel[3].iso_value, voxel[7].iso_value)
    end
    if (edges & 2048 == 2048)
        vertList[12] = edgeInterpolate(isolevel, voxel[4].vertex, voxel[8].vertex, voxel[4].iso_value, voxel[8].iso_value)
    end

    return vertList
end

function createTriangles(vertList, cubeIndex)
    trisLookup = triTable[cubeIndex]
    trisReal = collect(Iterators.filter(x -> x > -1, trisLookup))
    compactVertexArray = zeros(Float64, size(trisReal, 1) * 3)

    for (index, value) in enumerate(trisLookup)
        if value != -1
            compactVertexArray[1 + ((index - 1) * 3)] = vertList[value + 1].x
            compactVertexArray[2 + ((index - 1) * 3)] = vertList[value + 1].y
            compactVertexArray[3 + ((index - 1) * 3)] = vertList[value + 1].z
        end
    end
    # println(compactVertexArray)
    return compactVertexArray
end

function polygonise_ISO_SURFACE()
    iso_surface = fill_ISO_SURFACE(GRID_SIZE)
    vertexArray = Array{Float64}(undef, 0)

    for x = 1:size(iso_surface, 1) - 1
        for y = 1:size(iso_surface, 1) - 1
            for z = 1:size(iso_surface, 1) - 1
                voxel = [VoxelElement(Vertex(x, y, z), iso_surface[x, y, z]), VoxelElement(Vertex((x + 1), y, z), iso_surface[x + 1, y, z]),
                    VoxelElement(Vertex((x + 1), y, (z + 1)), iso_surface[x + 1, y, z + 1]), VoxelElement(Vertex(x, y, (z + 1)), iso_surface[x, y, z + 1]),
                    VoxelElement(Vertex(x, (y + 1), z), iso_surface[x, y + 1, z]), VoxelElement(Vertex((x + 1), (y + 1), z), iso_surface[x + 1, y + 1, z]),
                    VoxelElement(Vertex((x + 1), (y + 1), (z + 1)), iso_surface[x + 1, y + 1, z + 1]), VoxelElement(Vertex(x, (y + 1), (z + 1)), iso_surface[x, y + 1, z + 1])]


                cubeIndex = edge_lookup(voxel)
                if cubeIndex > 0
                    # println(edgeTable[cubeIndex + 1])    # julia is 1 indexed
                    vertList = vertices(edgeTable[cubeIndex + 1], voxel, ISO_VALUE)

                    # println(cubeIndex + 1)
                    # println(vertList)
                    singleVoxelVertexArray = createTriangles(vertList, cubeIndex + 1)
                    # println()
                    vertexArray = vcat(vertexArray, singleVoxelVertexArray)
                end
            end
        end
    end

    return vertexArray
end

function draw_ISO_SURFACE()
    # iso_surface = fill_ISO_SURFACE(GRID_SIZE)

    # volume(iso_surface, algorithm=:iso, isorange=ISO_RANGE, isovalue=ISO_VALUE)

    xyz = polygonise_ISO_SURFACE()

    x = xyz[1:3:end]
    y = xyz[2:3:end]
    z = xyz[3:3:end]

    mesh(x, y, z)

end

function check_ISO_SURFACE()
    iso_surface = fill_ISO_SURFACE(GRID_SIZE)
    println(iso_surface[1, 1, 1])

    println(iso_surface[1, 1, 2])

    println(iso_surface[1, 2, 1])

    println(iso_surface[2, 1, 1])

    println(iso_surface[13, 6, 21])

    println(iso_surface[1, 1, 17])

end

greet() = print("Hello World!")

# module MarchingSquares
# include("MarchingSquares.jl")
# end
# 
# module RenderGL
# include("RenderGL.jl")
# end
# 

end # module

