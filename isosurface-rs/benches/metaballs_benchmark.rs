use criterion::{black_box, criterion_group, criterion_main, Criterion};
use metaballs::iso_field_generator::{generate_iso_field, generate_iso_field2};
use ndarray::{Array, Array3};

const BALL_POS: [(f32, f32, f32); 2] = [(8.5, 8.5, 8.5), (8.5, 17.0, 8.5)];

const GRID_SIZE: usize = 128;

pub fn iso_surface_generator_benchmark(c: &mut Criterion) {
    c.bench_function("generate_iso_surface", |b| {
        b.iter(|| generate_iso_field(black_box(GRID_SIZE), black_box(&BALL_POS)))
    });
}

pub fn iso_surface_generator_benchmark2(c: &mut Criterion) {
    c.bench_function("generate_iso_surface2", |b| {
        let iso_surface: Array3<f32> = Array::zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE));
        b.iter(|| {
            generate_iso_field2(
                black_box(GRID_SIZE),
                black_box(&BALL_POS),
                black_box(&iso_surface),
            )
        })
    });
}

criterion_group!(
    benches,
    iso_surface_generator_benchmark,
    iso_surface_generator_benchmark2
);
criterion_main!(benches);
