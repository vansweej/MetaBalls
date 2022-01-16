use ndarray::{Array, Array3};
// use rayon::prelude::*;
// use rayon::vec::IntoIter;
use std::convert::From;
use std::ops::{Add, Mul};
use std::vec::IntoIter;
use crate::iso_field_generator::Voxel;
use num_traits::Zero;

type Indexes = (usize, usize, usize);
type Mask = (usize, usize, usize);

const MASKS: [Mask; 8] = [
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
struct Vertex1 {
    x: f32,
    y: f32,
    z: f32,
}

impl From<(Indexes, Mask)> for Vertex1 {
    fn from(i: (Indexes, Mask)) -> Self {
        Vertex1 {
            x: i.0.0.mul(2).add(i.1.0) as f32,
            y: i.0.1.mul(2).add(i.1.1) as f32,
            z: i.0.2.mul(2).add(i.1.2) as f32,
        }
    }
}

#[derive(Debug)]
struct VoxelElement1 {
    vertex: Vertex1,
    iso_value: Voxel,
}

type Voxel1 = [VoxelElement1; 8];

fn collect_iso_volume(iso_field: &Array3<Voxel>) -> Array3<Array3<Voxel>> {
    let grid_size = iso_field.dim().0;

    iso_field
        .exact_chunks((2, 2, 2))
        .into_iter()
        .map(|c| c.into_owned())
        .collect::<Array<Array3<Voxel>, _>>()
        .into_shape((grid_size / 2, grid_size / 2, grid_size / 2))
        .unwrap()
}

fn voxels_iter(iso_field: &Array3<Array3<Voxel>>) -> IntoIter<Voxel1> {
    let indexed_iter = iso_field.indexed_iter();
    indexed_iter
        .map(|(index, data)| {
            if index == (1, 1, 0) {
                println!("{:?}", index);
            }
            let voxel: Voxel1 = [
                {
                    let v = Vertex1 {
                        x: index.0.mul(2) as f32,
                        y: index.1.mul(2) as f32,
                        z: index.2.mul(2) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[0, 0, 0]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2).add(1) as f32,
                        y: index.1.mul(2) as f32,
                        z: index.2.mul(2) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[1, 0, 0]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2).add(1) as f32,
                        y: index.1.mul(2) as f32,
                        z: index.2.mul(2).add(1) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[1, 0, 1]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2) as f32,
                        y: index.1.mul(2) as f32,
                        z: index.2.mul(2).add(1) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[0, 0, 1]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2) as f32,
                        y: index.1.mul(2).add(1) as f32,
                        z: index.2.mul(2) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[0, 1, 0]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2).add(1) as f32,
                        y: index.1.mul(2).add(1) as f32,
                        z: index.2.mul(2) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[1, 1, 0]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2).add(1) as f32,
                        y: index.1.mul(2).add(1) as f32,
                        z: index.2.mul(2).add(1) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[1, 1, 1]],
                    }
                },
                {
                    let v = Vertex1 {
                        x: index.0.mul(2) as f32,
                        y: index.1.mul(2).add(1) as f32,
                        z: index.2.mul(2).add(1) as f32,
                    };
                    VoxelElement1 {
                        vertex: v,
                        iso_value: data[[0, 1, 1]],
                    }
                },
            ];
            voxel
        })
        .collect::<Vec<Voxel1>>()
        .into_iter()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::iso_field_generator::{generate_iso_field, ScalarField};
    use float_cmp::approx_eq;
    use pretty_assertions::assert_eq;

    const BALL_POS: [(f32, f32, f32); 2] = [(8.5, 8.5, 8.5), (8.5, 17.0, 8.5)];

    const GRID_SIZE: usize = 128;

    fn assert_voxel(voxel: &Voxel1, iso_cube: &ScalarField) {
        voxel.iter().for_each(|v| {
            let vertex_value = v.iso_value;
            let cube_value = iso_cube[[
                v.vertex.x as usize,
                v.vertex.y as usize,
                v.vertex.z as usize,
            ]];
            assert!(approx_eq!(Voxel, vertex_value, cube_value, ulps = 2));
        });
    }

    #[test]
    fn test_convert_iso_to_voxels() {
        let iso_cube = generate_iso_field(GRID_SIZE, &BALL_POS);

        let mut v_iter = voxels_iter(&collect_iso_volume(&iso_cube));
        //
        // //v_iter.for_each(|v| assert_voxel(&v, &iso_cube));
        //
        // println!();
        //
        let v1 = v_iter.nth(1).unwrap();
        assert_voxel(&v1, &iso_cube);
        // //
        // // println!();
        // //
        // // println!("{:?}", iso_cube[[0, 0, 1]]);
        //
        // let v2 = v_iter.nth(256).unwrap();
        // assert_voxel(&v2, &iso_cube);
    }
}
