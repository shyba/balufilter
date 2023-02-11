use std::hash::Hash;
use std::sync::{atomic::AtomicU64, Arc};

pub trait BaluFilter<T> where T: Hash {
    fn insert(&mut self, item: &T) -> bool;
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
    fn insert(&mut self, item: &T) -> bool {
        false
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
        assert_eq!(true, filter.insert(&"tchan"));
        assert_eq!(false, filter.insert(&"tchan"));
    }

}