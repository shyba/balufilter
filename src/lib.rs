use std::hash::{Hasher, Hash};
use std::sync::{atomic::AtomicU64, Arc};
use std::collections::hash_map::DefaultHasher;

pub trait BaluFilter<T> where T: Hash {
    fn insert(&self, item: &T) -> bool;
    fn check(&self, item: &T) -> bool;
}

pub struct AtomicFilter {
    contents: Arc<[AtomicU64; 32]>,
}

impl AtomicFilter {
    pub fn new() -> Self {
        const SLOT: AtomicU64 = AtomicU64::new(0);
        AtomicFilter {
            contents: Arc::new([SLOT; 32]),
        }
    }
}

impl<T: Hash> BaluFilter<T> for AtomicFilter {
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
        false
    }
}

#[cfg(test)]
mod tests {
    use crate::{BaluFilter, AtomicFilter};

    #[test]
    fn test_simple_insert() {
        let mut filter = AtomicFilter::new();
        assert_eq!(false, filter.insert(&"tchan"));
        assert_eq!(true, filter.insert(&"tchan"));
        assert_eq!(false, filter.insert(&"molejo"));
        assert_eq!(true, filter.insert(&"molejo"));
    }

}