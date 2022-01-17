use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use num_traits::Zero;
use std::convert::From;

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Voxel {
    iso_value: f32,
}

impl Voxel {
    pub fn new(value: f32) -> Voxel {
        Voxel {
            iso_value: value,
        }
    }
}

impl From<f32> for Voxel {
    fn from(f: f32) -> Self {
        Voxel {
            iso_value: f,
        }
    }
}

impl Add for Voxel {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self {
            iso_value: self.iso_value + rhs.iso_value,
        }
    }
}

impl AddAssign for Voxel {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = Self {
            iso_value: self.iso_value + rhs.iso_value,
        }
    }
}

impl Sub for Voxel {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self {
            iso_value: self.iso_value - rhs.iso_value,
        }
    }
}

impl SubAssign for Voxel {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = Self {
            iso_value: self.iso_value - rhs.iso_value,
        }
    }
}

impl Mul for Voxel {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self {
            iso_value: self.iso_value * rhs.iso_value,
        }
    }
}

impl MulAssign for Voxel {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self {
            iso_value: self.iso_value * rhs.iso_value,
        }
    }
}

impl Div for Voxel {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self {
            iso_value: self.iso_value / rhs.iso_value,
        }
    }
}

impl DivAssign for Voxel {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = Self {
            iso_value: self.iso_value / rhs.iso_value,
        }
    }
}

impl Zero for Voxel {
    #[inline]
    fn zero() -> Voxel {
        Voxel {
            iso_value: 0.0,
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.iso_value == 0.0
    }
}

#[cfg(test)]
mod tests {
    use float_cmp::{ApproxEq, F32Margin};
    use super::*;

    impl ApproxEq for Voxel {
        type Margin = F32Margin;

        fn approx_eq<T: Into<Self::Margin>>(self, other: Self, margin: T) -> bool {
            let margin = margin.into();

            self.iso_value.approx_eq(other.iso_value, margin)
        }
    }
}