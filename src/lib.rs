use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicU8;

pub trait BaluFilter<T, const N: usize, const K: usize>
where
    T: Hash,
{
    fn insert(&self, item: &T) -> bool;
    fn check(&self, item: &T) -> bool;
}

pub struct AtomicFilter<const N: usize, const K: usize> {
    contents: Vec<AtomicU8>,
}

impl<const N: usize, const K: usize> Default for AtomicFilter<N, K> {
    fn default() -> Self {
        AtomicFilter {
            contents: Vec::from_iter((0..N).into_iter().map(|_| AtomicU8::new(0))),
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

    pub fn bytes(&self) -> Vec<u8> {
        Vec::from_iter(
            self.contents
                .iter()
                .map(|v| v.load(std::sync::atomic::Ordering::SeqCst)),
        )
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
    use std::{
        collections::HashSet,
        hash::{Hash, Hasher},
        hint,
        sync::{atomic::AtomicU8, Arc},
    };

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
    fn test_very_large_filter_insert_does_not_blow_stack() {
        let filter: AtomicFilter<1_000_000, 50> = AtomicFilter::default();
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

    #[test]
    fn test_paralell_write() {
        let dataset = Arc::new(simple_sample_from_seed("parallel", 1_000_000));
        const THREADS: u8 = 8u8;
        let spinlock = Arc::new(AtomicU8::new(THREADS));

        const BYTES_SIZE: usize = 33_547_705 / 8;
        let parallel_filter: Arc<AtomicFilter<BYTES_SIZE, 23>> = Arc::new(AtomicFilter::default());
        for thread_index in 0..THREADS {
            let thread_spinlock = Arc::clone(&spinlock);
            let thread_filter = Arc::clone(&parallel_filter);
            let thread_dataset = Arc::clone(&dataset);
            std::thread::spawn(move || {
                for (index, value) in thread_dataset.iter().enumerate() {
                    if index % THREADS as usize == thread_index as usize {
                        thread_filter.insert(&value);
                    }
                }
                thread_spinlock.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
        }
        while spinlock.load(std::sync::atomic::Ordering::SeqCst) != 0 {
            hint::spin_loop();
        }

        let local_filter: AtomicFilter<BYTES_SIZE, 23> = AtomicFilter::default();
        for value in dataset.iter() {
            local_filter.insert(&value);
        }
        assert!(
            parallel_filter.bytes() == local_filter.bytes(),
            "filters mismatch"
        );
    }

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
}
