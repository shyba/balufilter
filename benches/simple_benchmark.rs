use std::collections::HashSet;

use balufilter::{AtomicFilter, BaluFilter};
use bloomfilter::Bloom;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let filter: AtomicFilter<480, 54> = AtomicFilter::default();
    c.bench_function("insert", |b| b.iter(|| filter.insert(&black_box("value"))));
    c.bench_function("check true", |b| {
        b.iter(|| filter.check(&black_box("value")))
    });
    c.bench_function("check false", |b| {
        b.iter(|| filter.check(&black_box("false")))
    });

    let mut table: HashSet<&str> = HashSet::default();
    c.bench_function("hashset insert", |b| {
        b.iter(|| table.insert(black_box("value")))
    });
    c.bench_function("hashset check true", |b| {
        b.iter(|| table.contains(black_box("value")))
    });
    c.bench_function("hashset check false", |b| {
        b.iter(|| table.contains(black_box("false")))
    });

    let mut bloomfilter = Bloom::new_with_seed(480, 50, &[0; 32]);
    println!("k: {}", bloomfilter.number_of_hash_functions());
    c.bench_function("bloomfilter insert", |b| {
        b.iter(|| bloomfilter.set(black_box("value")))
    });
    c.bench_function("bloomfilter check true", |b| {
        b.iter(|| bloomfilter.check(black_box("value")))
    });
    c.bench_function("bloomfilter check false", |b| {
        b.iter(|| bloomfilter.check(black_box("false")))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
