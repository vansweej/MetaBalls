use criterion::{black_box, criterion_group, criterion_main, Criterion};
use metaballs::iso_field_generator::{generate_iso_field, generate_iso_field2};
use ndarray::{Array, Array3};
use metaballs::compose;
use metaballs::compose::compose_two;
use metaballs::iso_field_polygoniser::{collect_iso_volume, cubes_iter, cubes_iter2};
use metaballs::voxel::Voxel;

const BALL_POS: [(f32, f32, f32); 2] = [(8.5, 8.5, 8.5), (8.5, 17.0, 8.5)];

const GRID_SIZE: usize = 256;

pub fn iso_surface_generator_benchmark(c: &mut Criterion) {
    c.bench_function("generate_iso_surface", |b| {
        b.iter(|| generate_iso_field(black_box(GRID_SIZE), black_box(&BALL_POS)))
    });
}

pub fn iso_surface_generator_benchmark2(c: &mut Criterion) {
    c.bench_function("generate_iso_surface2", |b| {
        let iso_surface: Array3<Voxel> = Array::zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE));
        b.iter(|| {
            generate_iso_field2(
                black_box(GRID_SIZE),
                black_box(&BALL_POS),
                black_box(&iso_surface),
            )
        })
    });
}

pub fn cubes_iter_benchmark1(c: &mut Criterion) {
    c.bench_function("cubes_iter", |b| {
        let iso_cube = generate_iso_field(GRID_SIZE, &BALL_POS);
        b.iter(||
            compose!(collect_iso_volume, cubes_iter)(&iso_cube)
        )
    });
}

pub fn cubes_iter_benchmark2(c: &mut Criterion) {
    c.bench_function("cubes_iter2", |b| {
        let iso_cube = generate_iso_field(GRID_SIZE, &BALL_POS);
        b.iter(||
            cubes_iter2(&iso_cube)
        )
    });
}

criterion_group!(
    benches,
    iso_surface_generator_benchmark,
    iso_surface_generator_benchmark2,
    cubes_iter_benchmark1,
    cubes_iter_benchmark2
);
criterion_main!(benches);
