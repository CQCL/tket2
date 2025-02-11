//! Pack lists of small ints into a bitvector using a unary encoding.
//!
//! Unary encoding stores 0 in 1bit, 1 in 2bits, etc. So it is very efficient
//! at encoding small integers.
//!
//! We use a flipped unary encoding, i.e. 0 -> 1, 1 -> 10, 2 -> 100, etc.
//! Thus the number of 1s in the encoding is the number of elements in the list,
//! and 0 is the empty list

use std::cmp;

use derive_more::{Display, Error};

/// A list of small ints packed into a `T`, typically u64.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, Default, serde::Serialize, serde::Deserialize,
)]
pub struct UnaryPacked<T> {
    data: T,
}

/// Trait for types that can be used as element type for [`UnaryPacked`].
///
/// This trait is currently implemented for `u64`, but could easily
/// be extended to other integer types.
pub trait BitField: Eq + Copy {
    fn n_bits() -> u32;

    fn zero() -> Self;

    fn shift_left(&mut self, n_bits: u32);

    fn shift_right(&mut self, n_bits: u32);

    fn set_bit(&mut self, bit: u32);

    fn count_ones(&self) -> u32;

    fn keep_lowest(&mut self, n_bits: u32);

    fn trailing_zeros(&self) -> u32;

    fn leading_zeros(&self) -> u32;
}

impl BitField for u64 {
    fn n_bits() -> u32 {
        64
    }

    fn zero() -> Self {
        0
    }

    fn shift_left(&mut self, n_bits: u32) {
        *self <<= n_bits;
    }

    fn shift_right(&mut self, n_bits: u32) {
        *self >>= n_bits;
    }

    fn set_bit(&mut self, bit: u32) {
        *self |= 1 << bit;
    }

    fn count_ones(&self) -> u32 {
        u64::count_ones(*self)
    }

    fn keep_lowest(&mut self, n_bits: u32) {
        *self &= !(1 << n_bits);
    }

    fn trailing_zeros(&self) -> u32 {
        u64::trailing_zeros(*self)
    }

    fn leading_zeros(&self) -> u32 {
        u64::leading_zeros(*self)
    }
}

/// An error that occurs when not enough bits are available.
#[derive(Error, Debug, Display)]
#[display("list is too long to encode in 64 bits")]
pub struct PackOverflow;

impl<T: BitField> UnaryPacked<T> {
    /// Create a new `UnaryPacked` from an iterator of numbers.
    pub fn try_from_iter(data: impl IntoIterator<Item = u32>) -> Result<Self, PackOverflow> {
        let mut list = Self::empty();
        for v in data {
            list.push_back(v)?;
        }
        Ok(list)
    }

    /// Create an empty `UnaryPacked`
    pub fn empty() -> Self {
        UnaryPacked { data: T::zero() }
    }

    /// Push a value to the back of the list
    pub fn push_back(&mut self, v: u32) -> Result<(), PackOverflow> {
        let n_bits = v + 1;
        if self.n_bits_used() + n_bits > T::n_bits() {
            return Err(PackOverflow);
        }

        self.data.shift_left(n_bits);
        self.data.set_bit(n_bits - 1);
        Ok(())
    }

    /// Get the number of elements in the list
    pub fn len(&self) -> u32 {
        self.data.count_ones()
    }

    /// Pop a value from the front of the list
    pub fn pop_front(&mut self) -> Option<u32> {
        let msb = self.n_bits_used().checked_sub(1)?;
        self.data.keep_lowest(msb);
        let Some(second_msb) = self.n_bits_used().checked_sub(1) else {
            return Some(msb);
        };
        Some(msb - second_msb - 1)
    }

    /// Pop a value from the back of the list
    pub fn pop_back(&mut self) -> Option<u32> {
        let lsb = self.data.trailing_zeros();
        if lsb == T::n_bits() {
            return None; // No set bits, list is empty
        }

        self.data.shift_right(lsb + 1);
        Some(lsb)
    }

    fn n_bits_used(&self) -> u32 {
        T::n_bits() - self.data.leading_zeros()
    }

    /// Convert the list to a vector
    #[allow(dead_code)]
    pub fn to_vec(self) -> Vec<u32> {
        let mut v = Vec::with_capacity(self.len() as usize);
        let mut list = self;
        while !list.is_empty() {
            v.push(list.pop_front().unwrap());
        }
        v
    }

    /// Check if the list is empty
    pub fn is_empty(&self) -> bool {
        self.data == T::zero()
    }
}

impl<T: BitField> TryFrom<&[u32]> for UnaryPacked<T> {
    type Error = PackOverflow;

    fn try_from(value: &[u32]) -> Result<Self, Self::Error> {
        UnaryPacked::try_from_iter(value.iter().copied())
    }
}

impl<T: BitField, const N: usize> TryFrom<&[u32; N]> for UnaryPacked<T> {
    type Error = PackOverflow;

    fn try_from(value: &[u32; N]) -> Result<Self, Self::Error> {
        UnaryPacked::try_from_iter(value.iter().copied())
    }
}

impl<T: BitField> PartialOrd for UnaryPacked<T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: BitField> Ord for UnaryPacked<T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        if self == other {
            return cmp::Ordering::Equal;
        }

        self.len()
            .cmp(&other.len())
            .then_with(|| element_wise_cmp(*self, *other))
    }
}

fn element_wise_cmp<T: BitField>(
    mut this: UnaryPacked<T>,
    mut other: UnaryPacked<T>,
) -> cmp::Ordering {
    // Assumes 1. length are equal 2. two arguments are not equal
    debug_assert_eq!(this.len(), other.len());
    debug_assert!(this != other);

    loop {
        let this_front = this.pop_front().unwrap();
        let other_front = other.pop_front().unwrap();
        match this_front.cmp(&other_front) {
            cmp::Ordering::Equal => continue,
            ordering => return ordering,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_list() {
        let mut list = UnaryPacked::<u64>::empty();

        list.push_back(2).unwrap();

        list.push_back(1).unwrap();

        list.push_back(0).unwrap();

        // The expected encoding should be:
        let exp = 0b0100101;

        assert_eq!(list.data, exp);

        assert_eq!(list.len(), 3);
    }

    #[test]
    fn test_push_list_overflow() {
        let mut list = UnaryPacked::<u64>::empty();

        // Try to push exactly MAX_BITS worth of data
        list.push_back(u64::n_bits() - 2).unwrap();
        list.push_back(0).unwrap();
        assert_eq!(list.data, (1 << (u64::n_bits() - 1)) + 1);

        list.push_back(0).unwrap_err();
    }

    #[test]
    fn test_pop_list() {
        let mut list = UnaryPacked::<u64>::try_from_iter([2, 1, 0, 2]).unwrap();

        assert_eq!(list.pop_front(), Some(2));
        assert_eq!(list.to_vec(), [1, 0, 2]);

        assert_eq!(list.pop_front(), Some(1));
        assert_eq!(list.to_vec(), [0, 2]);

        assert_eq!(list.pop_front(), Some(0));
        assert_eq!(list.to_vec(), [2]);

        assert_eq!(list.pop_front(), Some(2));
        assert!(list.is_empty());

        assert_eq!(list.pop_front(), None);
    }

    #[test]
    fn test_path_cmp() {
        let a: UnaryPacked<u64> = (&[1, 2]).try_into().unwrap();
        let b: UnaryPacked<u64> = (&[1, 2, 3]).try_into().unwrap();
        let c: UnaryPacked<u64> = (&[1, 3]).try_into().unwrap();

        assert!(a < b);
        assert_eq!(a, a);
        assert!(a < c);
    }
}
