use super::{
    super::functions::constants::{EPSILON, PRECISION},
    Vector,
};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct Vector3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}
impl Vector3 {
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
}
impl Vector for Vector3 {
    fn dim(self) -> usize {
        3
    }

    fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}
impl From<f32> for Vector3 {
    fn from(value: f32) -> Self {
        Self::new(value, value, value)
    }
}
impl Display for Vector3 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_len = [self.x, self.y, self.z]
            .iter()
            .map(|comp| (*comp as i32).to_string().len())
            .max()
            .unwrap();
        let space = " ".repeat(1 + max_len + 1 + PRECISION + 1);
        writeln!(f, "\n┌{space}┐").unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.x).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.y).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.z).unwrap();
        writeln!(f, "└{space}┘").unwrap();
        Ok(())
    }
}
impl Neg for Vector3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}
impl Add for Vector3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl AddAssign for Vector3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Vector3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl SubAssign for Vector3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Vector3 {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}
impl Mul<f32> for Vector3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl Mul<Vector3> for f32 {
    type Output = Vector3;

    fn mul(self, rhs: Vector3) -> Self::Output {
        rhs * self
    }
}
impl MulAssign<f32> for Vector3 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl Div<f32> for Vector3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Vector3 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}
impl PartialEq for Vector3 {
    fn eq(&self, other: &Self) -> bool {
        (*self - *other).len() < EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::constants::PI;

    // Constructors
    #[test]
    fn test1_vector3new() {
        let vec = Vector3 {
            x: 1.,
            y: 2.,
            z: 3.,
        };
        let vec_new = Vector3::new(1., 2., 3.);
        assert_eq!(vec, vec_new);
    }
    #[test]
    fn test1_vector3from() {
        let vec = Vector3 {
            x: 1.,
            y: 1.,
            z: 1.,
        };
        let vec_from = Vector3::from(1.);
        assert_eq!(vec, vec_from);
    }
    #[test]
    fn test1_vector3zero() {
        let vec = Vector3 {
            x: 0.,
            y: 0.,
            z: 0.,
        };
        let vec_from = Vector3::zero();
        assert_eq!(vec, vec_from);
    }
    // Impl Display
    #[test]
    fn test1_vector3display() {
        let vec = Vector3::from(PI);
        println!("{vec}");
    }
    // Impl PartialEq
    #[test]
    fn test1_vector3partial_eq() {
        let vec1 = Vector3::new(1., 7., 1.);
        let vec2 = Vector3::new(1., 8., 1.);
        assert_ne!(vec1, vec2);
    }
    #[test]
    fn test2_vector3partial_eq() {
        let vec1 = Vector3::new(2.3, 2.3, 2.3);
        let vec2 = Vector3::new(2.7, 2.7, 2.7);
        assert_ne!(vec1, vec2);
    }
    #[test]
    fn test3_vector3partial_eq() {
        let vec = Vector3::new(-1.5, 2.3, 1.);
        assert_eq!(vec, vec);
    }
    // Method dim()
    #[test]
    fn test1_vector3dim() {
        let dim = Vector3::default().dim();
        assert_eq!(dim, 3)
    }
    // Method len()
    #[test]
    fn test1_vector3len() {
        let vec = Vector3::new(6., 3., 2.);
        let len = vec.len();
        let exact_len = 7.;
        assert!((len - exact_len).abs() < EPSILON);
    }
    #[test]
    fn test2_vector3len() {
        let vec = Vector3::from(0.);
        let len = vec.len();
        assert!(len.abs() < EPSILON);
    }
    #[test]
    fn test3_vector3len() {
        let vec = Vector3::new(868.94, -1588.46, 354.26);
        let len = vec.len();
        let exact_len = 1844.92873;
        assert!((len - exact_len).abs() < EPSILON);
    }
    #[test]
    // Method normalize()
    fn test1_vector3normalize() {
        let vec = Vector3::from(3.);
        let vec_normalize = vec.normalize();
        let exact_vec_normalize = Vector3::from(1. / 3.0_f32.sqrt());
        assert_eq!(vec_normalize, exact_vec_normalize);
    }
    #[test]
    fn test2_vector3normalize() {
        let vec = Vector3::new(13., -7.5, 4.3);
        let vec_normalize = vec.normalize();
        let exact_vec_normalize = Vector3::new(0.832683, -0.480394, 0.275426);
        assert_eq!(vec_normalize, exact_vec_normalize);
    }
    #[test]
    #[should_panic]
    fn test3_vector3normalize() {
        Vector3::zero().normalize();
    }
    // Method angle(vector)
    #[test]
    fn test1_vector3angle() {
        let vec1 = Vector3::new(1., 0., 0.);
        let vec2 = Vector3::new(0., 1., 0.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test2_vector3angle() {
        let vec1 = Vector3::new(1., 0., 0.);
        let vec2 = Vector3::new(0., 0., 1.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test3_vector3angle() {
        let vec1 = Vector3::new(1., 0., -1.);
        let vec2 = Vector3::new(0., -1., 0.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test4_vector3angle() {
        let vec1 = Vector3::new(2., 2., 3.);
        let vec2 = Vector3::new(6., 6., 9.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        assert!(angle1 < EPSILON && angle2 < EPSILON);
    }
    #[test]
    fn test5_vector3angle() {
        let vec1 = Vector3::new(-3., 3., 6.);
        let vec2 = Vector3::new(1., -1., -2.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test6_vector3angle() {
        let vec1 = Vector3::new(1., 2., 9.);
        let vec2 = Vector3::new(7., 8., -7.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = 1.91653;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    #[should_panic]
    fn test7_vector3angle() {
        Vector3::new(1., 1., 1.).angle(Vector3::zero());
    }
    #[test]
    #[should_panic]
    fn test8_vector3angle() {
        Vector3::zero().angle(Vector3::new(1., 1., 1.));
    }
    // Method is_orthogonal_to(vector)
    #[test]
    fn test1_vector2is_orthogonal_to() {
        let vec1 = Vector3::new(1., 0., 0.);
        let vec2 = Vector3::new(0., -7., 5.);
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test2_vector2is_orthogonal_to() {
        let vec1 = Vector3::new(6., -2., 2.);
        let vec2 = Vector3::new(1., 4., 1.);
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test3_vector2is_orthogonal_to() {
        let vec1 = Vector3::new(1., 4., -17.);
        let vec2 = Vector3::zero();
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test4_vector2is_orthogonal_to() {
        let vec = Vector3::zero();
        assert!(vec.is_orthogonal_to(vec));
    }
    #[test]
    fn test5_vector2is_orthogonal_to() {
        let vec1 = Vector3::new(1., 4., 1.);
        let vec2 = Vector3::new(7., -4., 0.5);
        assert_ne!(vec1.is_orthogonal_to(vec2), true);
        assert_ne!(vec2.is_orthogonal_to(vec1), true);
    }
    // Method cross(vector)
    #[test]
    fn test1_vector3cross() {
        let vec1 = Vector3::new(1., 0., 0.);
        let vec2 = Vector3::new(0., 0., 2.);
        let cross_vec = vec1.cross(vec2);
        assert!(cross_vec.is_orthogonal_to(vec1));
        assert!(cross_vec.is_orthogonal_to(vec2));
    }
    #[test]
    fn test2_vector3cross() {
        let vec1 = Vector3::new(1., 13., -1.);
        let vec2 = Vector3::new(2., -5., 2.);
        let cross_vec = vec1.cross(vec2);
        assert!(cross_vec.is_orthogonal_to(vec1));
        assert!(cross_vec.is_orthogonal_to(vec2));
    }
    #[test]
    fn test3_vector3cross() {
        let vec1 = Vector3::new(1., 13., -1.);
        let vec2 = Vector3::zero();
        let cross_vec = vec1.cross(vec2);
        assert_eq!(cross_vec, Vector3::zero());
        assert!(cross_vec.is_orthogonal_to(vec1));
        assert!(cross_vec.is_orthogonal_to(vec2));
    }
    #[test]
    fn test4_vector3cross() {
        let vec1 = Vector3::new(1., 13., -1.);
        let vec2 = vec1 * 4.;
        let cross_vec = vec1.cross(vec2);
        assert_eq!(cross_vec, Vector3::zero());
        assert!(cross_vec.is_orthogonal_to(vec1));
        assert!(cross_vec.is_orthogonal_to(vec2));
    }
    #[test]
    fn test5_vector3cross() {
        let vec = Vector3::new(1., 13., -1.);
        let cross_vec = vec.cross(vec);
        assert_eq!(cross_vec, Vector3::zero());
        assert!(cross_vec.is_orthogonal_to(vec));
    }
    // Impl Neg
    #[test]
    fn test1_vector3neg() {
        let vec = Vector3::new(-1., 3., 4.);
        let neg_vec = Vector3::new(1., -3., -4.);
        assert_eq!(-vec, neg_vec)
    }
    #[test]
    fn test2_vector3neg() {
        let vec = Vector3::zero();
        assert_eq!(-vec, vec);
    }
    #[test]
    fn test3_vector3neg() {
        let vec = Vector3::new(1., 7., -2.);
        assert_eq!(--vec, vec);
    }
    // Impl Add
    #[test]
    fn test1_vector3add() {
        let vec1 = Vector3::new(1.5, -2., 1.);
        let vec2 = Vector3::new(-7.8, 4., -1.);
        let add_vec = Vector3::new(-6.3, 2., 0.);
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, add_vec)
    }
    #[test]
    fn test2_vector3add() {
        let vec1 = Vector3::new(1.5, -2., 1.);
        let vec2 = Vector3::zero();
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, vec1);
    }
    #[test]
    fn test3_vector3add() {
        let vec = Vector3::zero();
        assert_eq!(vec + vec, vec);
    }
    #[test]
    fn test4_vector3add() {
        let vec1 = Vector3::new(1., -3., 0.);
        let vec2 = -vec1;
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, Vector3::zero());
    }
    // Impl AddAsiign
    #[test]
    fn test1_vector3add_assign() {
        let mut vec1 = Vector3::new(1.5, -2., 1.);
        let vec2 = Vector3::new(-7.8, 4., -1.);
        let add_assign_vec = vec1 + vec2;
        vec1 += vec2;
        assert_eq!(vec1, add_assign_vec);
    }
    #[test]
    fn test2_vector3add_assign() {
        let mut vec1 = Vector3::new(1.5, -2., 1.);
        let vec2 = Vector3::zero();
        let add_assign_vec = vec1;
        vec1 += vec2;
        assert_eq!(vec1, add_assign_vec);
    }
    #[test]
    fn test3_vector3add_assign() {
        let mut vec = Vector3::zero();
        vec += vec;
        assert_eq!(vec, Vector3::zero());
    }
    #[test]
    fn test4_vector3add_assign() {
        let mut vec1 = Vector3::new(1., -3., 0.);
        let vec2 = -vec1;
        vec1 += vec2;
        assert_eq!(vec1, Vector3::zero());
    }
    // Impl Sub
    #[test]
    fn test1_vector3sub() {
        let vec1 = Vector3::new(1.5, -2., 4.);
        let vec2 = Vector3::new(-7.8, 4., 4.);
        let sub_vec = Vector3::new(9.3, -6., 0.);
        assert_eq!(vec1 - vec2, -(vec2 - vec1));
        assert_eq!(vec1 - vec2, sub_vec)
    }
    #[test]
    fn test2_vector3sub() {
        let vec1 = Vector3::new(1.5, -2., 4.);
        let vec2 = Vector3::zero();
        assert_eq!(vec1 - vec2, -(vec2 - vec1));
        assert_eq!(vec1 - vec2, vec1);
    }
    #[test]
    fn test3_vector3sub() {
        let vec = Vector3::new(1., -3., 0.);
        assert_eq!(vec - vec, vec - vec);
        assert_eq!(vec - vec, Vector3::zero());
    }
    // Impl SubAssign
    #[test]
    fn test1_vector3sub_assign() {
        let mut vec1 = Vector3::new(1.5, -2., 4.);
        let vec2 = Vector3::new(-7.8, 4., 4.);
        let sub_assign_vec = vec1 - vec2;
        vec1 -= vec2;
        assert_eq!(vec1, sub_assign_vec);
    }
    #[test]
    fn test2_vector3sub_assign() {
        let mut vec1 = Vector3::new(1.5, -2., 4.);
        let vec2 = Vector3::zero();
        let sub_assign_vec = vec1;
        vec1 -= vec2;
        assert_eq!(vec1, sub_assign_vec);
    }
    #[test]
    fn test3_vector3sub_assign() {
        let mut vec = Vector3::new(1., -3., 0.);
        vec -= vec;
        assert_eq!(vec, Vector3::zero());
    }
    // Impl Mul
    #[test]
    fn test1_vector3mul() {
        let vec1 = Vector3::new(1., 4., 2.);
        let vec2 = Vector3::new(-7., 3., -1.);
        let exact_mul = 3.;
        assert_eq!(vec1 * vec2, vec2 * vec1);
        assert_eq!(vec1 * vec2, exact_mul);
    }
    #[test]
    fn test2_vector3mul() {
        let vec1 = Vector3::new(17.3, -5., 1.2);
        let vec2 = Vector3::zero();
        assert_eq!(vec1 * vec2, vec2 * vec1);
        assert_eq!(vec1 * vec2, 0.);
    }
    // Impl Mul<f32>
    #[test]
    fn test1_vector3mul_f32() {
        let vec = Vector3::new(1., -4., 0.1);
        let value = 13.;
        let mul_vec = Vector3::new(13., -52., 1.3);
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, mul_vec);
    }
    #[test]
    fn test2_vector3mul_f32() {
        let vec = Vector3::new(1., -4., 0.1);
        let value = 2.;
        let mul_vec = vec + vec;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, mul_vec);
    }
    #[test]
    fn test3_vector3mul_f32() {
        let vec = Vector3::new(1., -4., 0.1);
        let value = 1.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, vec);
    }
    #[test]
    fn test4_vector3mul_f32() {
        let vec = Vector3::new(1., -4., 0.1);
        let value = 0.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, Vector3::zero());
    }
    #[test]
    fn test5_vector3mul_f32() {
        let vec = Vector3::new(1., -4., 0.1);
        let value = -1.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, -vec);
    }
    // Impl MulAssign<f32>
    #[test]
    fn test1_vector3mul_assign_f32() {
        let mut vec = Vector3::new(1., -4., 0.1);
        let value = 13.;
        let mul_assign_vec = vec * value;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test2_vector3mul_assign_f32() {
        let mut vec = Vector3::new(1., -4., 0.1);
        let value = 2.;
        let mul_assign_vec = vec + vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test3_vector3mul_assign_f32() {
        let mut vec = Vector3::new(1., -4., 0.1);
        let value = 1.;
        let mul_assign_vec = vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test4_vector3mul_assign_f32() {
        let mut vec = Vector3::new(1., -4., 0.1);
        let value = 0.;
        vec *= value;
        assert_eq!(vec, Vector3::zero());
    }
    #[test]
    fn test5_vector3mul_assign_f32() {
        let mut vec = Vector3::new(1., -4., 0.1);
        let value = -1.;
        let mul_assign_vec = -vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    // Impl Div<f32>
    #[test]
    fn test1_vector3div_f32() {
        let vec = Vector3::new(17.4, 9.3, -3.);
        let value = 3.;
        let div_vec = Vector3::new(5.8, 3.1, -1.);
        assert_eq!(vec / value, div_vec)
    }
    #[test]
    fn test2_vector3div_f32() {
        let vec = Vector3::new(17.4, 9.3, -3.);
        let value = 1.;
        assert_eq!(vec / value, vec);
    }
    #[test]
    fn test3_vector3div_f32() {
        let vec = Vector3::new(17.4, 9.3, -3.);
        let value = -1.;
        assert_eq!(vec / value, -vec);
    }
    // Impl DivAssign<f32>
    #[test]
    fn test1_vector3div_assign_f32() {
        let mut vec = Vector3::new(17.4, 9.3, -3.);
        let value = 3.;
        let div_assign_vec = vec / value;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
    #[test]
    fn test2_vector3div_assign_f32() {
        let mut vec = Vector3::new(17.4, 9.3, -3.);
        let value = 1.;
        let div_assign_vec = vec;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
    #[test]
    fn test3_vector3div_assign_f32() {
        let mut vec = Vector3::new(17.4, 9.3, -3.);
        let value = -1.;
        let div_assign_vec = -vec;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
}
