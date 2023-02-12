use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{atomic::AtomicU64, Arc};

pub trait BaluFilter<T, const N: usize, const K: usize>
where
    T: Hash,
{
    fn insert(&self, item: &T) -> bool;
    fn check(&self, item: &T) -> bool;
}

pub struct AtomicFilter<const N: usize, const K: usize> {
    contents: Arc<[AtomicU64]>,
}

impl<const N: usize, const K: usize> Default for AtomicFilter<N, K> {
    fn default() -> Self {
        AtomicFilter {
            contents: Arc::new(std::array::from_fn::<AtomicU64, N, _>(|_| {
                AtomicU64::new(0)
            })),
        }
    }
}

impl<const N: usize, const K: usize> AtomicFilter<N, K> {
    #[inline(always)]
    fn operation<T: Hash, const WRITE: bool>(&self, item: &T) -> bool {
        let mut hasher = DefaultHasher::new();
        let mut was_there = true;
        for _ in 0..K {
            item.hash(&mut hasher);
            let bit_index = hasher.finish() % 64;
            let shift = 1 << bit_index;
            let prev = if WRITE {
                self.contents[(bit_index % N as u64) as usize]
                    .fetch_or(shift, std::sync::atomic::Ordering::SeqCst)
            } else {
                self.contents[(bit_index % N as u64) as usize]
                    .load(std::sync::atomic::Ordering::Relaxed)
            };
            was_there &= (prev & shift) == shift;
        }
        was_there
    }
}

impl<T: Hash, const N: usize, const K: usize> BaluFilter<T, N, K> for AtomicFilter<N, K> {
    fn insert(&self, item: &T) -> bool {
        self.operation::<T, true>(item)
    }

    fn check(&self, item: &T) -> bool {
        self.operation::<T, false>(item)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtomicFilter, BaluFilter};

    #[test]
    fn test_simple_insert() {
        let filter: AtomicFilter<320, 17> = AtomicFilter::default();
        assert_eq!(false, filter.insert(&"tchan"));
        assert_eq!(true, filter.insert(&"tchan"));
        assert_eq!(false, filter.insert(&"molejo"));
        assert_eq!(true, filter.insert(&"molejo"));
    }

    #[test]
    fn test_insert_check() {
        let filter: AtomicFilter<30, 43> = AtomicFilter::default();
        assert_eq!(false, filter.check(&"tchan"));
        assert_eq!(false, filter.insert(&"tchan"));
        assert_eq!(true, filter.check(&"tchan"));
        assert_eq!(true, filter.insert(&"tchan"));

        assert_eq!(false, filter.check(&"molejo"));
        assert_eq!(false, filter.insert(&"molejo"));
        assert_eq!(true, filter.check(&"molejo"));
        assert_eq!(true, filter.insert(&"molejo"));
    }
}
