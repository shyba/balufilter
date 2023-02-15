use balufilter::{AtomicFilter, BaluFilter};
use bloomfilter::Bloom;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

use highway::{self, HighwayBuildHasher};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut cycle = (0..1_000_000).cycle();
    let mut inverse_cycle = (10_000_000..11_000_000).cycle();

    const BYTES_SIZE: usize = 33_547_705 / 8;
    let filter: AtomicFilter<BYTES_SIZE, 23> = AtomicFilter::default();
    c.bench_function("insert", |b| {
        b.iter(|| filter.insert(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("check true", |b| {
        b.iter(|| filter.check(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("check false", |b| {
        b.iter(|| filter.check(&black_box(inverse_cycle.next().unwrap())))
    });

    let filter: AtomicFilter<BYTES_SIZE, 23, HighwayBuildHasher> = AtomicFilter::with_state(highway::HighwayBuildHasher::default());
    c.bench_function("highway insert", |b| {
        b.iter(|| filter.insert(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("highway check true", |b| {
        b.iter(|| filter.check(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("highway check false", |b| {
        b.iter(|| filter.check(&black_box(inverse_cycle.next().unwrap())))
    });

    let mut filter = cuckoofilter::CuckooFilter::new();
    c.bench_function("cuckoo insert", |b| {
        b.iter(|| filter.add(black_box(&cycle.next().unwrap())))
    });
    c.bench_function("cuckoo check true", |b| {
        b.iter(|| filter.contains(black_box(&cycle.next().unwrap())))
    });
    c.bench_function("cuckoo check false", |b| {
        b.iter(|| filter.contains(black_box(&inverse_cycle.next().unwrap())))
    });

    let mut bloomfilter = Bloom::new_for_fp_rate(1_000_000, 0.0000001);
    println!(
        "k: {}, m: {}",
        bloomfilter.number_of_hash_functions(),
        bloomfilter.number_of_bits()
    );
    c.bench_function("bloomfilter insert", |b| {
        b.iter(|| bloomfilter.set(black_box(&cycle.next().unwrap())))
    });
    c.bench_function("bloomfilter check true", |b| {
        b.iter(|| bloomfilter.check(black_box(&cycle.next().unwrap())))
    });
    c.bench_function("bloomfilter check false", |b| {
        b.iter(|| bloomfilter.check(black_box(&inverse_cycle.next().unwrap())))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
