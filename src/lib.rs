#![no_std]

use core as std;
extern crate alloc;
use alloc::vec::Vec;
use std::hash::{BuildHasher, Hash, Hasher};
use std::sync::atomic::AtomicU8;

#[cfg(feature = "ahasher")]
use ahash::RandomState;
#[cfg(feature = "random")]
use rand::random;

pub trait BaluFilter<T, const N: usize, const K: usize>
where
    T: Hash,
{
    fn insert(&self, item: &T) -> bool;
    fn check(&self, item: &T) -> bool;
}

pub struct AtomicFilter<const N: usize, const K: usize, B: BuildHasher> {
    contents: Vec<AtomicU8>,
    hash_builder: B,
    seed: u64,
}

impl<const N: usize, const K: usize, B: BuildHasher> AtomicFilter<N, K, B> {
    pub fn with_state_and_seed(state: B, seed: u64) -> Self {
        let mut hasher = state.build_hasher();
        seed.hash(&mut hasher);
        AtomicFilter {
            contents: std::iter::repeat_with(|| AtomicU8::new(0))
                .take(N)
                .collect(),
            hash_builder: state,
            seed: hasher.finish(),
        }
    }
}

#[cfg(feature = "ahasher")]
impl<const N: usize, const K: usize> AtomicFilter<N, K, RandomState> {
    fn with_seed(seed: u64) -> Self {
        AtomicFilter::with_state_and_seed(RandomState::with_seed(seed as usize), seed)
    }
}

#[cfg(all(feature = "random", feature = "ahasher"))]
impl<const N: usize, const K: usize> Default for AtomicFilter<N, K, RandomState> {
    fn default() -> Self {
        AtomicFilter::with_seed(random())
    }
}

impl<const N: usize, const K: usize, B: BuildHasher> AtomicFilter<N, K, B> {
    #[inline]
    fn modulo(value: u32, n: u32) -> u32 {
        // http://lemire.me/blog/2016/06/27/a-fast-alternative-to-the-modulo-reduction/
        (((value as u64) * (n as u64)) >> 32) as u32
    }
    #[inline(always)]
    fn operation<T: Hash, const WRITE: bool>(&self, item: &T) -> bool {
        let mut hasher = self.hash_builder.build_hasher();
        let mut was_there = true;
        item.hash(&mut hasher);
        let mut hash = hasher.finish().wrapping_add(self.seed);
        for round in 1..=(K / 2) {
            if !self.check_round::<WRITE>(hash as u32, &mut was_there) && !WRITE {
                return false;
            }
            if !self.check_round::<WRITE>((hash.rotate_left(32)) as u32, &mut was_there) && !WRITE {
                return false;
            }
            hash = hash.wrapping_add(hash.rotate_left(round as u32));
        }
        was_there
    }
    #[inline(always)]
    fn check_round<const WRITE: bool>(&self, hash: u32, was_there: &mut bool) -> bool {
        let shift = 1 << (hash & 0x7);
        let byte_index = Self::modulo(hash, N as u32) as usize;
        let prev = self.contents[byte_index].load(std::sync::atomic::Ordering::Relaxed);
        *was_there &= (prev & shift) == shift;
        if WRITE {
            self.contents[byte_index].fetch_or(shift, std::sync::atomic::Ordering::Relaxed);
        }
        *was_there
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
    extern crate std;
    use std::sync::Arc;

    use crate::{AtomicFilter, BaluFilter};
    use ahash::RandomState;
    use alloc::vec;

    #[test]
    fn test_simple_insert() {
        let filter: AtomicFilter<320, 50, RandomState> = AtomicFilter::default();
        assert_eq!(false, filter.insert(&"tchan"));
        assert_eq!(true, filter.insert(&"tchan"));
        assert_eq!(false, filter.insert(&"molejo"));
        assert_eq!(true, filter.insert(&"molejo"));
    }

    #[test]
    fn test_very_large_filter_insert_does_not_blow_stack() {
        const BYTES_SIZE: usize = 33_547_705 / 8;
        let filter: AtomicFilter<BYTES_SIZE, 23, RandomState> = AtomicFilter::default();
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
        for value in 1_000_000..101_000_000 {
            if filter.check(&value) {
                p += 1.0f64;
            }
        }
        p /= 100_000_000f64;
        assert!(p < 0.0000002f64 as f64, "P = {} > 0.0000002", p);

        for value in 0..1_000_000 {
            assert!(filter.check(&value));
        }
    }

    #[test]
    fn test_insert_check() {
        let filter: AtomicFilter<320, 43, RandomState> = AtomicFilter::default();
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

        const BYTES_SIZE: usize = 33_547_705 / 8;
        let parallel_filter: Arc<AtomicFilter<BYTES_SIZE, 23, RandomState>> =
            Arc::new(AtomicFilter::with_seed(42));
        let mut handles = vec![];
        for thread_index in 0..THREADS {
            let thread_filter = Arc::clone(&parallel_filter);
            handles.push(std::thread::spawn(move || {
                for index in 0..1_000_000 {
                    if index % THREADS as usize == thread_index as usize {
                        thread_filter.insert(&index);
                    }
                }
            }));
        }
        let local_filter: AtomicFilter<BYTES_SIZE, 23, RandomState> = AtomicFilter::with_seed(42);
        for value in 0..1_000_000 {
            local_filter.insert(&value);
        }
        for handle in handles {
            handle.join().unwrap();
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

        const BYTES_SIZE: usize = 33_547_705 / 8;
        let parallel_filter: Arc<AtomicFilter<BYTES_SIZE, 23, RandomState>> =
            Arc::new(AtomicFilter::with_seed(42));
        // Readers start first as they won't leave until things match, so they can wait for writers
        let mut handles = vec![];
        for _ in 0..READERS {
            let thread_filter = Arc::clone(&parallel_filter);
            handles.push(std::thread::spawn(move || {
                let mut running = true;
                while running {
                    let mut p = 0f64;
                    for element in 10_000_000..11_000_000 {
                        if thread_filter.check(&element) {
                            p += 1f64;
                        }
                    }
                    p /= 1_000_000f64;
                    assert!(p < 0.000002f64 as f64, "P = {} > 0.000002", p);
                    let mut found = false;
                    for element in 0..1_000_000 {
                        if !thread_filter.check(&element) {
                            found = true;
                        }
                    }
                    running = found;
                }
            }));
        }
        for thread_index in 0..WRITERS {
            let thread_filter = Arc::clone(&parallel_filter);
            handles.push(std::thread::spawn(move || {
                for index in 0..1_000_000 {
                    if index % WRITERS as usize == thread_index as usize {
                        thread_filter.insert(&index);
                    }
                }
            }));
        }
        let local_filter: AtomicFilter<BYTES_SIZE, 23, RandomState> = AtomicFilter::with_seed(42);
        for value in 0..1_000_000 {
            local_filter.insert(&value);
        }

        for handle in handles {
            handle.join().unwrap();
        }

        assert!(
            parallel_filter.bytes() == local_filter.bytes(),
            "filters mismatch"
        );
    }
}
