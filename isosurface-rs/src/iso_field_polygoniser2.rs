use ndarray::{Array, Array3};
// use rayon::prelude::*;
// use rayon::vec::IntoIter;
use std::convert::From;
use std::ops::{Add, Mul};
use std::vec::IntoIter;

const MASKS: [(usize, usize, usize); 8] = [
    (0, 0, 0),
    (1, 0, 0),
    (1, 0, 1),
    (0, 0, 1),
    (0, 1, 0),
    (1, 1, 0),
    (1, 1, 1),
    (0, 1, 1),
];

#[derive(Debug)]
struct Vertex {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Debug)]
struct VoxelElement {
    vertex: Vertex,
    iso_value: f32,
}

#[derive(Debug)]
struct VoxelCube {
    voxel: [VoxelElement; 8],
}

#[derive(Debug)]
struct VoxelSpace {
    voxel_space: Array3<VoxelCube>,
}

fn collect_iso_volume(iso_field: &Array3<f32>) -> Array3<Array3<f32>> {
    let grid_size = iso_field.dim().0;

    iso_field
        .exact_chunks((2, 2, 2))
        .into_iter()
        .map(|c| c.into_owned())
        .collect::<Array<Array3<f32>, _>>()
        .into_shape((grid_size / 2, grid_size / 2, grid_size / 2))
        .unwrap()
}

// impl From<&Array3<f32>> for VoxelSpace {
//     fn from(iso_field: &Array3<f32>) -> Self {
//         let iso_space_as_array = collect_iso_volume(&iso_field);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iso_field_generator::generate_iso_field;

    const BALL_POS: [(f32, f32, f32); 2] = [(8.5, 8.5, 8.5), (8.5, 17.0, 8.5)];

    const GRID_SIZE: usize = 128;

    #[test]
    fn test_transforms_to_voxelspace() {
        let iso_field = generate_iso_field(GRID_SIZE, &BALL_POS);

        //        let x = VoxelSpace::from(&iso_field);
    }
}
