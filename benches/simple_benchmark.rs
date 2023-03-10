mod no_hash;
use balufilter::{blocking::BlockedAtomicFilter, AtomicFilter, BaluFilter};
use bloomfilter::Bloom;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use no_hash::NoHasher;
use std::hash::BuildHasher;

use ahash::RandomState;
use highway::{self, HighwayBuildHasher};
use rustc_hash::FxHasher;

struct FxBuilder {}
impl BuildHasher for FxBuilder {
    type Hasher = FxHasher;

    fn build_hasher(&self) -> Self::Hasher {
        FxHasher::default()
    }
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut cycle = (0..1_000_000).cycle();
    let mut inverse_cycle = (10_000_000..11_000_000).cycle();

    const BITS_SIZE: usize = 33_547_705;
    let filter: BlockedAtomicFilter<56_547_705, 24, RandomState> = BlockedAtomicFilter::default();
    c.bench_function("register blocked insert", |b| {
        b.iter(|| filter.insert(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("register blocked check true", |b| {
        b.iter(|| filter.check(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("register blocked check false", |b| {
        b.iter(|| filter.check(&black_box(inverse_cycle.next().unwrap())))
    });
    let filter: AtomicFilter<BITS_SIZE, 23, NoHasher> =
        AtomicFilter::with_state_and_seed(NoHasher::new(), 42);
    c.bench_function("nohash insert", |b| {
        b.iter(|| filter.insert(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("nohash check true", |b| {
        b.iter(|| filter.check(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("nohash check false", |b| {
        b.iter(|| filter.check(&black_box(inverse_cycle.next().unwrap())))
    });
    let filter: AtomicFilter<BITS_SIZE, 23, FxBuilder> =
        AtomicFilter::with_state_and_seed(FxBuilder {}, 42);
    c.bench_function("fx insert", |b| {
        b.iter(|| filter.insert(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("fx check true", |b| {
        b.iter(|| filter.check(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("fx check false", |b| {
        b.iter(|| filter.check(&black_box(inverse_cycle.next().unwrap())))
    });
    let filter: AtomicFilter<BITS_SIZE, 23, RandomState> = AtomicFilter::default();
    c.bench_function("insert", |b| {
        b.iter(|| filter.insert(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("check true", |b| {
        b.iter(|| filter.check(&black_box(cycle.next().unwrap())))
    });
    c.bench_function("check false", |b| {
        b.iter(|| filter.check(&black_box(inverse_cycle.next().unwrap())))
    });

    let filter: AtomicFilter<BITS_SIZE, 23, HighwayBuildHasher> =
        AtomicFilter::with_state_and_seed(highway::HighwayBuildHasher::default(), 42);
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
