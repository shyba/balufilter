use criterion::{black_box, criterion_group, criterion_main, Criterion};
use balufilter::{BaluFilter, AtomicFilter};

pub fn criterion_benchmark(c: &mut Criterion) {
    let filter: AtomicFilter<480, 43> = AtomicFilter::default();
    c.bench_function("batata", |b| b.iter(|| filter.insert(&black_box("batata"))));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);