using MetaBalls:MetaBalls

function generate_voxels(field_size)
    iso_surface = MetaBalls.fill_ISO_SURFACE(field_size)
    voxelArray = Array{Voxel}(undef, 0)

    for x in 1:size(iso_surface, 1) - 1
        for y in 1:size(iso_surface, 2) - 1
            for z in 1:size(iso_surface, 3) - 1
                voxel::Voxel = [VoxelElement(Vertex(x, y, z), iso_surface[x,y,z]),                          VoxelElement(Vertex((x + 1), y, z), iso_surface[x + 1, y, z]), 
                         VoxelElement(Vertex((x + 1), y, (z + 1)), iso_surface[x + 1, y, z + 1]),           VoxelElement(Vertex(x, y, (z + 1)), iso_surface[x, y, z + 1]),
                         VoxelElement(Vertex(x, (y + 1), z), iso_surface[x,y + 1,z]),                       VoxelElement(Vertex((x + 1), (y + 1), z), iso_surface[x + 1, y + 1, z]), 
                         VoxelElement(Vertex((x + 1), (y + 1), (z + 1)), iso_surface[x + 1, y + 1, z + 1]), VoxelElement(Vertex(x, (y + 1), (z + 1)), iso_surface[x, y + 1, z + 1])]

                push!(voxelArray, voxel)
            end
        end
    end

    return voxelArray
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