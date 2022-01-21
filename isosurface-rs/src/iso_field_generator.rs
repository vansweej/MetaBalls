use std::cell::RefCell;
use ndarray::{Array, Array3, IntoDimension, Zip};
use partial_application::partial;
use ndarray::azip;
//use rayon::prelude::IntoParallelIterator;

use crate::voxel::Voxel;

#[inline(always)]
fn calculate_voxel_value(
    grid_pos: (usize, usize, usize),
    ball_positions: &[(f32, f32, f32)],
) -> Voxel {
    Voxel::new(ball_positions
        .iter()
        .map(|(posx, posy, posz)| {
            1.0 / (f32::powi(posx - grid_pos.0 as f32, 2)
                + f32::powi(posy - grid_pos.1 as f32, 2)
                + f32::powi(posz - grid_pos.2 as f32, 2))
        })
        .sum())
}

pub type ScalarField = Array3<Voxel>;

pub fn generate_iso_field(grid_size: usize, ball_positions: &[(f32, f32, f32)]) -> ScalarField {
    let voxel_value = partial!(calculate_voxel_value => _, ball_positions);

    Array::from_shape_fn([grid_size; 3].into_dimension(), voxel_value)
}

pub fn generate_iso_field2(
    grid_size: usize,
    ball_positions: &[(f32, f32, f32)],
    iso_surface: &ScalarField,
) -> ScalarField {
    iso_surface
        .indexed_iter()
        .map(|x: ((usize, usize, usize), &Voxel)| calculate_voxel_value(x.0, ball_positions))
        .collect::<Array<_, _>>()
        .into_shape((grid_size, grid_size, grid_size))
        .unwrap()
}

pub struct CachedScalarField {
    index_cache: Array3<(usize, usize, usize)>,
    scalar_field: RefCell<Array3<Voxel>>,
}

impl CachedScalarField {
    pub fn new(size: usize) -> Self {
        CachedScalarField {
            index_cache: Array::from_shape_fn([size; 3].into_dimension(), |(x, y, z)| (x, y, z)),
            scalar_field: RefCell::new(Array::zeros((size, size, size))),
        }
    }

    pub fn generate_iso_field(&mut self, ball_positions: &[(f32, f32, f32)]) {

        let r = self.scalar_field.get_mut();

        Zip::from(r).and(&self.index_cache).for_each(|sf, &b| {*sf = calculate_voxel_value(b, ball_positions)});
    }

    pub fn generate_iso_field_mt(&mut self, ball_positions: &[(f32, f32, f32)]) {

        let r = self.scalar_field.get_mut();

        Zip::from(r).and(&self.index_cache).par_for_each(|sf, &b| {*sf = calculate_voxel_value(b, ball_positions)});
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::BorrowMut;
    use super::*;
    use float_cmp::{approx_eq};
    use pretty_assertions::assert_eq;

    pub const BALL_POS: [(f32, f32, f32); 2] = [(8.5, 8.5, 8.5), (8.5, 17.0, 8.5)];

    pub const GRID_SIZE: usize = 32;

    #[test]
    fn test_generate_iso_surface() {
        let result = generate_iso_field(GRID_SIZE, &BALL_POS);

        assert_eq!(result.shape(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]);
    }

    #[test]
    fn test_generate_iso_surface2() {
        let iso_surface: ScalarField = Array::zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE));
        let result = generate_iso_field2(GRID_SIZE, &BALL_POS, &iso_surface);

        assert_eq!(result.shape(), [GRID_SIZE, GRID_SIZE, GRID_SIZE]);
    }

    #[test]
    fn test_compare_two_generators() {
        let result1 = generate_iso_field(GRID_SIZE, &BALL_POS);

        let iso_surface: ScalarField = Array::zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE));
        let result2 = generate_iso_field2(GRID_SIZE, &BALL_POS, &iso_surface);

        result1
            .iter()
            .zip(result2.iter())
            .for_each(|(a, b)| assert!(approx_eq!(Voxel, *a, *b, ulps = 2)));
    }

    #[test]
    fn test_caching_generator() {
        let mut test = CachedScalarField::new(GRID_SIZE);
        test.index_cache.indexed_iter().for_each(|(index, data)| {assert_eq!(index.0, data.0); assert_eq!(index.1, data.1); assert_eq!(index.2, data.2)});

        test.generate_iso_field(&BALL_POS);

        let test1 = generate_iso_field(GRID_SIZE, &BALL_POS);

        test.scalar_field.borrow().indexed_iter().for_each(|(index, data)| {assert!(approx_eq!(Voxel, *data, test1[[index.0, index.1, index.2]], ulps = 2))});
    }

    #[test]
    fn test_caching_generator_mt() {
        let mut test = CachedScalarField::new(GRID_SIZE);
        test.index_cache.indexed_iter().for_each(|(index, data)| {assert_eq!(index.0, data.0); assert_eq!(index.1, data.1); assert_eq!(index.2, data.2)});

        test.generate_iso_field_mt(&BALL_POS);

        let test1 = generate_iso_field(GRID_SIZE, &BALL_POS);

        test.scalar_field.borrow().indexed_iter().for_each(|(index, data)| {assert!(approx_eq!(Voxel, *data, test1[[index.0, index.1, index.2]], ulps = 2))});
    }

}
