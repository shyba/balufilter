use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{atomic::AtomicU8, Arc};

pub trait BaluFilter<T, const N: usize, const K: usize>
where
    T: Hash,
{
    fn insert(&self, item: &T) -> bool;
    fn check(&self, item: &T) -> bool;
}

pub struct AtomicFilter<const N: usize, const K: usize> {
    contents: Arc<[AtomicU8]>,
}

impl<const N: usize, const K: usize> Default for AtomicFilter<N, K> {
    fn default() -> Self {
        AtomicFilter {
            contents: Arc::new(std::array::from_fn::<AtomicU8, N, _>(|_| AtomicU8::new(0))),
        }
    }
}

impl<const N: usize, const K: usize> AtomicFilter<N, K> {
    #[inline(always)]
    fn operation<T: Hash, const WRITE: bool>(&self, item: &T) -> bool {
        let mut hasher = DefaultHasher::new();
        let mut was_there = true;
        item.hash(&mut hasher);
        let mut hash = hasher.finish();
        let multiplier = hash >> 32;
        let mask = N as u64 - 1;
        for round in 0..K {
            let shift = 1 << (hash & 0x7);
            let byte_index = ((hash >> 8) & mask) as usize;
            let prev = if WRITE {
                self.contents[byte_index].fetch_or(shift, std::sync::atomic::Ordering::SeqCst)
            } else {
                self.contents[byte_index].load(std::sync::atomic::Ordering::SeqCst)
            };
            was_there &= (prev & shift) == shift;
            if !was_there && !WRITE {
                return false;
            }
            hash = hash.wrapping_add(round as u64).wrapping_mul(multiplier);
        }
        was_there
    }
}

impl<T: Hash, const N: usize, const K: usize> BaluFilter<T, N, K> for AtomicFilter<N, K> {
    #[inline]
    fn insert(&self, item: &T) -> bool {
        self.operation::<T, true>(item)
    }

    #[inline]
    fn check(&self, item: &T) -> bool {
        self.operation::<T, false>(item)
    }
}

#[cfg(test)]
mod tests {
    use crate::{AtomicFilter, BaluFilter};

    #[test]
    fn test_simple_insert() {
        let filter: AtomicFilter<320, 50> = AtomicFilter::default();
        assert_eq!(false, filter.insert(&"tchan"));
        assert_eq!(true, filter.insert(&"tchan"));
        assert_eq!(false, filter.insert(&"molejo"));
        assert_eq!(true, filter.insert(&"molejo"));
    }

    #[test]
    fn test_insert_check() {
        let filter: AtomicFilter<320, 43> = AtomicFilter::default();
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
