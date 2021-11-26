# All code in here are directly derived from the C code

using MetaBalls:MetaBalls

ISO_VALUE = 0.1

function edge_lookup(voxel)
    cubeIndex = 0
    
    # println(ISO_MIN, " ", voxel[1].value, " ", ISO_MAX)

    # if (ISO_MIN < voxel[1].value) && (voxel[1].value < ISO_MAX) cubeIndex |= 1 end
    # if (ISO_MIN < voxel[2].value) && (voxel[2].value < ISO_MAX) cubeIndex |= 2 end
    # if (ISO_MIN < voxel[3].value) && (voxel[3].value < ISO_MAX) cubeIndex |= 4 end
    # if (ISO_MIN < voxel[4].value) && (voxel[4].value < ISO_MAX) cubeIndex |= 8 end
    # if (ISO_MIN < voxel[5].value) && (voxel[5].value < ISO_MAX) cubeIndex |= 16 end
    # if (ISO_MIN < voxel[6].value) && (voxel[6].value < ISO_MAX) cubeIndex |= 32 end
    # if (ISO_MIN < voxel[7].value) && (voxel[7].value < ISO_MAX) cubeIndex |= 64 end
    # if (ISO_MIN < voxel[8].value) && (voxel[8].value < ISO_MAX) cubeIndex |= 128 end

    if (voxel[1].iso_value < ISO_VALUE) cubeIndex |= 1 end
    if (voxel[2].iso_value < ISO_VALUE) cubeIndex |= 2 end
    if (voxel[3].iso_value < ISO_VALUE) cubeIndex |= 4 end
    if (voxel[4].iso_value < ISO_VALUE) cubeIndex |= 8 end
    if (voxel[5].iso_value < ISO_VALUE) cubeIndex |= 16 end
    if (voxel[6].iso_value < ISO_VALUE) cubeIndex |= 32 end
    if (voxel[7].iso_value < ISO_VALUE) cubeIndex |= 64 end
    if (voxel[8].iso_value < ISO_VALUE) cubeIndex |= 128 end

    return cubeIndex
end