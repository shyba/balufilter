use ahash::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::AtomicU8;

pub trait BaluFilter<T, const N: usize, const K: usize>
where
    T: Hash,
{
    fn insert(&self, item: &T) -> bool;
    fn check(&self, item: &T) -> bool;
}

pub struct AtomicFilter<const N: usize, const K: usize, B = RandomState> {
    contents: Vec<AtomicU8>,
    hash_builder: B,
}

impl<const N: usize, const K: usize> AtomicFilter<N, K, RandomState> {
    fn with_state(state: RandomState) -> Self {
        AtomicFilter {
            contents: std::iter::repeat_with(|| AtomicU8::new(0))
                .take(N)
                .collect(),
            hash_builder: state,
        }
    }

    fn with_seed(seed: usize) -> Self {
        AtomicFilter::with_state(RandomState::with_seed(seed))
    }
}

impl<const N: usize, const K: usize> Default for AtomicFilter<N, K, RandomState> {
    fn default() -> Self {
        AtomicFilter::with_state(RandomState::new())
    }
}

impl<const N: usize, const K: usize, B: BuildHasher> AtomicFilter<N, K, B> {
    #[inline(always)]
    fn operation<T: Hash, const WRITE: bool>(&self, item: &T) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        let mut was_there = true;
        item.hash(&mut hasher);
        let mut hash = hasher.finish();
        for round in 0..K {
            let shift = 1 << (hash & 0x7);
            let byte_index = hash as usize % N;
            let prev = if WRITE {
                self.contents[byte_index].fetch_or(shift, std::sync::atomic::Ordering::SeqCst)
            } else {
                self.contents[byte_index].load(std::sync::atomic::Ordering::SeqCst)
            };
            was_there &= (prev & shift) == shift;
            if !was_there && !WRITE {
                return false;
            }
            hasher.write_u64(hash.wrapping_add(round as u64));
            hash = hasher.finish();
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

impl<T: Hash, const N: usize, const K: usize, B: BuildHasher> BaluFilter<T, N, K>
    for AtomicFilter<N, K, B>
{
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
        const BYTES_SIZE: usize = 33_547_705 / 8;
        let filter: AtomicFilter<BYTES_SIZE, 23> = AtomicFilter::default();
        dbg!(filter.bytes().len());
        for value in 0..1_000_000 {
            assert!(!filter.check(&value));
        }
        for value in 0..1_000_000 {
            filter.insert(&value);
        }
        for value in 0..1_000_000 {
            assert!(filter.check(&value));
        }
        let mut p = 0.0f64;
        for value in 10_000_000..11_000_000 {
            if filter.check(&value) {
                p += 1.0f64;
            }
        }
        p /= 1_000_000f64;
        assert!(p < 0.0001f64 as f64, "P = {} > 0.0001", p);

        for value in 0..1_000_000 {
            assert!(filter.check(&value));
        }
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
        const THREADS: u8 = 8u8;
        let spinlock = Arc::new(AtomicU8::new(THREADS));

        const BYTES_SIZE: usize = 33_547_705 / 8;
        let parallel_filter: Arc<AtomicFilter<BYTES_SIZE, 23>> =
            Arc::new(AtomicFilter::with_seed(42));
        for thread_index in 0..THREADS {
            let thread_spinlock = Arc::clone(&spinlock);
            let thread_filter = Arc::clone(&parallel_filter);
            std::thread::spawn(move || {
                for index in 0..1_000_000 {
                    if index % THREADS as usize == thread_index as usize {
                        thread_filter.insert(&index);
                    }
                }
                thread_spinlock.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
        }
        let local_filter: AtomicFilter<BYTES_SIZE, 23> = AtomicFilter::with_seed(42);
        for value in 0..1_000_000 {
            local_filter.insert(&value);
        }
        while spinlock.load(std::sync::atomic::Ordering::SeqCst) != 0 {
            hint::spin_loop();
        }
        assert!(
            parallel_filter.bytes() == local_filter.bytes(),
            "filters mismatch"
        );
    }

    #[test]
    fn test_paralell_read_while_writing() {
        // Given:
        // dataset = data that will be added to the bloom filter
        // inverse_dataset = data that should not match
        // Goal:
        // Writers: Each writer will write a shard (1/THREADS) of the matching dataset
        // Readers: will process the whole inverse dataset ensuring the inverse never matches until the matching dataset matches entirely
        const READERS: u8 = 2;
        const WRITERS: u8 = 2;
        const THREADS: u8 = READERS + WRITERS;
        let spinlock = Arc::new(AtomicU8::new(THREADS));

        const BYTES_SIZE: usize = 33_547_705 / 8;
        let parallel_filter: Arc<AtomicFilter<BYTES_SIZE, 23>> =
            Arc::new(AtomicFilter::with_seed(42));
        // Readers start first as they won't leave until things match, so they can wait for writers
        for _ in 0..READERS {
            let thread_spinlock = Arc::clone(&spinlock);
            let thread_filter = Arc::clone(&parallel_filter);
            std::thread::spawn(move || {
                let mut running = true;
                while running {
                    for element in 10_000_000..11_000_000 {
                        if thread_filter.check(&element) {
                            thread_spinlock.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                            panic!("Hit an element from the inverse dataset")
                        }
                    }
                    let mut found = false;
                    for element in 0..1_000_000 {
                        if !thread_filter.check(&element) {
                            found = true;
                        }
                    }
                    running = found;
                }
                thread_spinlock.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
        }
        for thread_index in 0..WRITERS {
            let thread_spinlock = Arc::clone(&spinlock);
            let thread_filter = Arc::clone(&parallel_filter);
            std::thread::spawn(move || {
                for index in 0..1_000_000 {
                    if index % WRITERS as usize == thread_index as usize {
                        thread_filter.insert(&index);
                    }
                }
                thread_spinlock.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            });
        }
        let local_filter: AtomicFilter<BYTES_SIZE, 23> = AtomicFilter::with_seed(42);
        for value in 0..1_000_000 {
            local_filter.insert(&value);
        }

        while spinlock.load(std::sync::atomic::Ordering::SeqCst) != 0 {
            hint::spin_loop();
        }

        assert!(
            parallel_filter.bytes() == local_filter.bytes(),
            "filters mismatch"
        );
    }
}
