use std::hash::{Hasher, BuildHasher};

pub struct NoHasher {
    state: u64
}

impl NoHasher {
    pub fn new() -> Self {
        NoHasher { state: 42 }
    }
}

impl Hasher for NoHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.state
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for val in bytes {
            self.state += *val as u64;
        }
    }
}

impl BuildHasher for NoHasher {
    type Hasher = NoHasher;

    #[inline]
    fn build_hasher(&self) -> Self::Hasher {
        NoHasher::new()
    }
}