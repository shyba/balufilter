use std::{
    collections::HashSet,
    hash::{Hash, Hasher},
};

use balufilter::{AtomicFilter, BaluFilter};
use bloomfilter::Bloom;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn simple_sample_from_seed(seed: &str, size: usize) -> HashSet<String> {
    sample_from_seed_excluding(seed, size, &HashSet::new())
}

pub fn sample_from_seed_excluding(
    seed: &str,
    size: usize,
    exclude: &HashSet<String>,
) -> HashSet<String> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    seed.hash(&mut hasher);
    let mut set = HashSet::new();
    let mut index = 0;
    while set.len() < size {
        index.hash(&mut hasher);
        let value = format!("{:x}", hasher.finish());
        if !exclude.contains(&value) {
            set.insert(format!("{:x}", hasher.finish()));
        }
        index += 1;
    }
    set
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut cycle = (0..1_000_000).cycle();
    let mut inverse_cycle = (10_000_000..11_000_000).cycle();

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
