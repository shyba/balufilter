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
            contents: Arc::new(std::array::from_fn::<AtomicU64, N, _>(|_| AtomicU64::new(0))),
        }
    }
}

impl<T: Hash, const N: usize, const K: usize> BaluFilter<T, N, K> for AtomicFilter<N, K> {
    fn insert(&self, item: &T) -> bool {
        let mut hasher = DefaultHasher::new();
        let mut was_there = false;
        for index in 0..32 {
            item.hash(&mut hasher);
            let hash = hasher.finish();
            let prev = self.contents[index].fetch_or(hash, std::sync::atomic::Ordering::AcqRel);
            was_there |= (prev & hash) == hash;
        }
        was_there
    }

    fn check(&self, item: &T) -> bool {
        let mut hasher = DefaultHasher::new();
        let mut was_there = false;
        for index in 0..K {
            item.hash(&mut hasher);
            let hash = hasher.finish();
            let prev = self.contents[index % N].load(std::sync::atomic::Ordering::Relaxed);
            was_there |= (prev & hash) == hash;
        }
        was_there
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtomicFilter, BaluFilter};

    #[test]
    fn test_simple_insert() {
        let filter: AtomicFilter<32, 30> = AtomicFilter::default();
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
