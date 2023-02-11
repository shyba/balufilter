use std::sync::{atomic::AtomicU64, Arc};

pub trait BaluFilter {
    type Item;
    fn insert(&mut self, item: &Self::Item) -> bool;
    fn check(&self, item: &Self::Item) -> bool;
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

#[cfg(test)]
mod tests {

}