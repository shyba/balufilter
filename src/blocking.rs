use core::sync::atomic::Ordering::Relaxed;
use core::{
    hash::{BuildHasher, Hash},
    sync::atomic::AtomicUsize,
};

#[cfg(feature = "ahasher")]
use ahash::RandomState;

#[cfg(feature = "random")]
use rand::random;

use alloc::vec::Vec;

use crate::BaluFilter;

const ATOMIC_USIZE_SIZE_IN_BYTES: usize = (usize::BITS / 8) as usize;
const CACHE_LINE_BYTES: usize = 64;
const BLOCK_SIZE: usize = CACHE_LINE_BYTES / ATOMIC_USIZE_SIZE_IN_BYTES;

type Block = [AtomicUsize; BLOCK_SIZE];

#[repr(C, align(512))]
pub struct BlockedAtomicFilter<const M: usize, const K: usize, B: BuildHasher> {
    contents: Vec<Block>,
    hash_builder: B,
    seed: u64,
}

impl<const M: usize, const K: usize, B: BuildHasher> BlockedAtomicFilter<M, K, B> {
    pub fn with_state_and_seed(state: B, seed: u64) -> Self {
        let blocks = core::cmp::max(1, ((M / 8) / ATOMIC_USIZE_SIZE_IN_BYTES) / BLOCK_SIZE);
        let seed_hash = state.hash_one(seed);
        BlockedAtomicFilter {
            contents: core::iter::repeat_with(|| core::array::from_fn(|_| AtomicUsize::new(0)))
                .take(blocks)
                .collect(),
            hash_builder: state,
            seed: seed_hash,
        }
    }
}

#[cfg(feature = "ahasher")]
impl<const M: usize, const K: usize> BlockedAtomicFilter<M, K, RandomState> {
    fn with_seed(seed: u64) -> Self {
        BlockedAtomicFilter::with_state_and_seed(RandomState::with_seed(seed as usize), seed)
    }
}

#[cfg(all(feature = "random", feature = "ahasher"))]
impl<const M: usize, const K: usize> Default for BlockedAtomicFilter<M, K, RandomState> {
    fn default() -> Self {
        BlockedAtomicFilter::with_seed(random())
    }
}

impl<T: Hash, const M: usize, const K: usize, B: BuildHasher> BaluFilter<T, M, K>
    for BlockedAtomicFilter<M, K, B>
{
    #[inline]
    fn insert(&self, item: &T) -> bool {
        let mut hash = self.hash_builder.hash_one(item) ^ self.seed;
        let block_index = hash as usize % self.contents.len();
        let block = &self.contents[block_index];
        let mut found = true;

        let batches = K / block.len();
        assert!(batches > 0 && batches <= 8);
        for (index, sector) in block.iter().enumerate() {
            let mut search_mask = 0;
            for _ in 0..batches {
                hash = hash.rotate_left(8);
                search_mask |= 1 << ((hash & 0xff) as u8 % usize::BITS as u8);
            }
            found &= sector.fetch_or(search_mask, Relaxed) & search_mask == search_mask;
            hash = hash.wrapping_add(1 + hash.rotate_left(1 + index as u32));
        }
        found
    }

    #[inline]
    fn check(&self, item: &T) -> bool {
        let mut hash = self.hash_builder.hash_one(item) ^ self.seed;
        let block_index = hash as usize % self.contents.len();
        let block = &self.contents[block_index];
        let mut found = true;

        let batches = K / block.len();
        assert!(batches > 0 && batches <= 8);
        for (index, sector) in block.iter().enumerate() {
            let mut search_mask = 0;
            for _ in 0..batches {
                hash = hash.rotate_left(8);
                search_mask |= 1 << ((hash & 0xff) as u8 % usize::BITS as u8);
            }
            found &= sector.load(Relaxed) & search_mask == search_mask;
            if !found {
                return false;
            }
            hash = hash.wrapping_add(1 + hash.rotate_left(1 + index as u32));
        }
        found
    }
}

#[cfg(test)]
mod test {
    extern crate std;

    use crate::BaluFilter;
    use ahash::RandomState;

    use super::BlockedAtomicFilter;

    #[test]
    fn test_blocked_insert_check() {
        let filter: BlockedAtomicFilter<320, 8, RandomState> = BlockedAtomicFilter::default();
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
    fn test_very_large_blocked_filter_insert_does_not_blow_stack() {
        let filter: BlockedAtomicFilter<56_547_705, 24, RandomState> =
            BlockedAtomicFilter::default();
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
        for value in 1_000_000..11_000_000 {
            if filter.check(&value) {
                p += 1.0f64;
            }
        }
        p /= 10_000_000f64;
        assert!(p < 0.0000002f64 as f64, "P = {} > 0.0000002", p); // f-rate is worse

        for value in 0..1_000_000 {
            assert!(filter.check(&value));
        }
    }
}
