use balufilter::{AtomicFilter, BaluFilter};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(c: &mut Criterion) {
    let filter: AtomicFilter<480, 43> = AtomicFilter::default();
    c.bench_function("insert", |b| b.iter(|| filter.insert(&black_box("insert"))));
    c.bench_function("check", |b| b.iter(|| filter.check(&black_box("check"))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
