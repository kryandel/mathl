use super::{
    super::functions::constants::{EPSILON, PRECISION},
    Vector,
};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct Vector4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}
impl Vector4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }
}
impl Vector for Vector4 {
    fn dim(self) -> usize {
        4
    }

    fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w).sqrt()
    }
}
impl From<f32> for Vector4 {
    fn from(value: f32) -> Self {
        Self::new(value, value, value, value)
    }
}
impl Display for Vector4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_len = [self.x, self.y, self.z, self.w]
            .iter()
            .map(|comp| (*comp as i32).to_string().len())
            .max()
            .unwrap();
        let space = " ".repeat(1 + max_len + 1 + PRECISION + 1);
        writeln!(f, "\n┌{space}┐").unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.x).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.y).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.z).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.w).unwrap();
        writeln!(f, "└{space}┘").unwrap();
        Ok(())
    }
}
impl Neg for Vector4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}
impl Add for Vector4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}
impl AddAssign for Vector4 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Vector4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl SubAssign for Vector4 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Vector4 {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}
impl Mul<f32> for Vector4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}
impl Mul<Vector4> for f32 {
    type Output = Vector4;

    fn mul(self, rhs: Vector4) -> Self::Output {
        rhs * self
    }
}
impl MulAssign<f32> for Vector4 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs
    }
}
impl Div<f32> for Vector4 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Vector4 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}
impl PartialEq for Vector4 {
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
    fn test1_vector4new() {
        let vec = Vector4 {
            x: 1.,
            y: 2.,
            z: 3.,
            w: 4.,
        };
        let vec_new = Vector4::new(1., 2., 3., 4.);
        assert_eq!(vec, vec_new);
    }
    #[test]
    fn test1_vector4from() {
        let vec = Vector4 {
            x: 1.,
            y: 1.,
            z: 1.,
            w: 1.,
        };
        let vec_from = Vector4::from(1.);
        assert_eq!(vec, vec_from);
    }
    #[test]
    fn test1_vector4zero() {
        let vec = Vector4 {
            x: 0.,
            y: 0.,
            z: 0.,
            w: 0.,
        };
        let vec_from = Vector4::zero();
        assert_eq!(vec, vec_from);
    }
    // Impl Display
    #[test]
    fn test1_vector4display() {
        let vec = Vector4::from(PI);
        println!("{vec}");
    }
    // Impl PartialEq
    #[test]
    fn test1_vector4partial_eq() {
        let vec1 = Vector4::new(1., 7., 1., 0.);
        let vec2 = Vector4::new(1., 8., 1., 0.);
        assert_ne!(vec1, vec2);
    }
    #[test]
    fn test2_vector4partial_eq() {
        let vec1 = Vector4::new(2.3, 2.3, 2.3, 2.3);
        let vec2 = Vector4::new(2.7, 2.7, 2.7, 2.7);
        assert_ne!(vec1, vec2);
    }
    #[test]
    fn test3_vector4partial_eq() {
        let vec = Vector4::new(-1.5, 2.3, 1., -4.2);
        assert_eq!(vec, vec);
    }
    // Method dim()
    #[test]
    fn test1_vector4dim() {
        let dim = Vector4::default().dim();
        assert_eq!(dim, 4)
    }
    // Method len()
    #[test]
    fn test1_vector4len() {
        let vec = Vector4::new(5., 2., 4., 2.);
        let len = vec.len();
        let exact_len = 7.;
        assert!((len - exact_len).abs() < EPSILON);
    }
    #[test]
    fn test2_vector4len() {
        let vec = Vector4::from(0.);
        let len = vec.len();
        assert!(len.abs() < EPSILON);
    }
    #[test]
    fn test3_vector4len() {
        let vec = Vector4::new(868.94, -1588.46, 354.26, -10.1);
        let len = vec.len();
        let exact_len = 1844.95638;
        assert!((len - exact_len).abs() < EPSILON);
    }
    #[test]
    // Method normalize()
    fn test1_vector4normalize() {
        let vec = Vector4::from(3.);
        let vec_normalize = vec.normalize();
        let exact_vec_normalize = Vector4::from(0.5);
        assert_eq!(vec_normalize, exact_vec_normalize);
    }
    #[test]
    fn test2_vector4normalize() {
        let vec = Vector4::new(13., -7.5, 4.3, 0.2);
        let vec_normalize = vec.normalize();
        let exact_vec_normalize = Vector4::new(0.832615, -0.480355, 0.275403, 0.0128095);
        assert_eq!(vec_normalize, exact_vec_normalize);
    }
    #[test]
    #[should_panic]
    fn test3_vector4normalize() {
        Vector4::zero().normalize();
    }
    // Method angle(vector)
    #[test]
    fn test1_vector4angle() {
        let vec1 = Vector4::new(1., 0., 0., 1.);
        let vec2 = Vector4::new(0., 1., 0., 1.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 3.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test2_vector4angle() {
        let vec1 = Vector4::new(1., 0., 0., 0.);
        let vec2 = Vector4::new(0., 0., 1., -1.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test3_vector4angle() {
        let vec1 = Vector4::new(1., 0., -1., -1.);
        let vec2 = Vector4::new(0., -1., 0., 0.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test4_vector4angle() {
        let vec1 = Vector4::new(2., 2., 3., 1.);
        let vec2 = Vector4::new(6., 6., 9., 3.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        assert!(angle1 < EPSILON && angle2 < EPSILON);
    }
    #[test]
    fn test5_vector4angle() {
        let vec1 = Vector4::new(-3., 3., 6., 9.3);
        let vec2 = Vector4::new(1., -1., -2., -3.1);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test6_vector4angle() {
        let vec1 = Vector4::new(1., 2., 9., 0.);
        let vec2 = Vector4::new(7., 8., -7., 1.7);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = 1.91336;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    #[should_panic]
    fn test7_vector4angle() {
        Vector4::new(1., 1., 1., 1.).angle(Vector4::zero());
    }
    #[test]
    #[should_panic]
    fn test8_vector4angle() {
        Vector4::zero().angle(Vector4::new(1., 1., 1., 1.));
    }
    // Method is_orthogonal_to(vector)
    #[test]
    fn test1_vector2is_orthogonal_to() {
        let vec1 = Vector4::new(1., 0., 0., -3.);
        let vec2 = Vector4::new(0., -7., 5., 0.);
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test2_vector2is_orthogonal_to() {
        let vec1 = Vector4::new(6., -2., 2., 0.);
        let vec2 = Vector4::new(1., 4., 1., 3.);
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test3_vector2is_orthogonal_to() {
        let vec1 = Vector4::new(1., 4., -17., 1.);
        let vec2 = Vector4::zero();
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test4_vector2is_orthogonal_to() {
        let vec = Vector4::zero();
        assert!(vec.is_orthogonal_to(vec));
    }
    #[test]
    fn test5_vector2is_orthogonal_to() {
        let vec1 = Vector4::new(1., 4., 1., 13.17);
        let vec2 = Vector4::new(7., -4., 0.5, -7.);
        assert_ne!(vec1.is_orthogonal_to(vec2), true);
        assert_ne!(vec2.is_orthogonal_to(vec1), true);
    }
    // Impl Neg
    #[test]
    fn test1_vector4neg() {
        let vec = Vector4::new(-1., 3., 4., -11.1);
        let neg_vec = Vector4::new(1., -3., -4., 11.1);
        assert_eq!(-vec, neg_vec)
    }
    #[test]
    fn test2_vector4neg() {
        let vec = Vector4::zero();
        assert_eq!(-vec, vec);
    }
    #[test]
    fn test3_vector4neg() {
        let vec = Vector4::new(1., 7., -2., 0.);
        assert_eq!(--vec, vec);
    }
    // Impl Add
    #[test]
    fn test1_vector4add() {
        let vec1 = Vector4::new(1.5, -2., 1., 1.3);
        let vec2 = Vector4::new(-7.8, 4., -1., -0.3);
        let add_vec = Vector4::new(-6.3, 2., 0., 1.);
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, add_vec)
    }
    #[test]
    fn test2_vector4add() {
        let vec1 = Vector4::new(1.5, -2., 1., 1.3);
        let vec2 = Vector4::zero();
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, vec1);
    }
    #[test]
    fn test3_vector4add() {
        let vec = Vector4::zero();
        assert_eq!(vec + vec, vec);
    }
    #[test]
    fn test4_vector4add() {
        let vec1 = Vector4::new(1., -3., 0., 5.);
        let vec2 = -vec1;
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, Vector4::zero());
    }
    // Impl AddAsiign
    #[test]
    fn test1_vector4add_assign() {
        let mut vec1 = Vector4::new(1.5, -2., 1., 1.3);
        let vec2 = Vector4::new(-7.8, 4., -1., -0.3);
        let add_assign_vec = vec1 + vec2;
        vec1 += vec2;
        assert_eq!(vec1, add_assign_vec);
    }
    #[test]
    fn test2_vector4add_assign() {
        let mut vec1 = Vector4::new(1.5, -2., 1., 1.3);
        let vec2 = Vector4::zero();
        let add_assign_vec = vec1;
        vec1 += vec2;
        assert_eq!(vec1, add_assign_vec);
    }
    #[test]
    fn test3_vector4add_assign() {
        let mut vec = Vector4::zero();
        vec += vec;
        assert_eq!(vec, Vector4::zero());
    }
    #[test]
    fn test4_vector4add_assign() {
        let mut vec1 = Vector4::new(1., -3., 0., 5.);
        let vec2 = -vec1;
        vec1 += vec2;
        assert_eq!(vec1, Vector4::zero());
    }
    // Impl Sub
    #[test]
    fn test1_vector4sub() {
        let vec1 = Vector4::new(1.5, -2., 4., 1.);
        let vec2 = Vector4::new(-7.8, 4., 4., 0.);
        let sub_vec = Vector4::new(9.3, -6., 0., 1.);
        assert_eq!(vec1 - vec2, -(vec2 - vec1));
        assert_eq!(vec1 - vec2, sub_vec)
    }
    #[test]
    fn test2_vector4sub() {
        let vec1 = Vector4::new(1.5, -2., 4., 1.);
        let vec2 = Vector4::zero();
        assert_eq!(vec1 - vec2, -(vec2 - vec1));
        assert_eq!(vec1 - vec2, vec1);
    }
    #[test]
    fn test3_vector4sub() {
        let vec = Vector4::new(1., -3., 0., 5.);
        assert_eq!(vec - vec, vec - vec);
        assert_eq!(vec - vec, Vector4::zero());
    }
    // Impl SubAssign
    #[test]
    fn test1_vector4sub_assign() {
        let mut vec1 = Vector4::new(1.5, -2., 4., 1.);
        let vec2 = Vector4::new(-7.8, 4., 4., 0.);
        let sub_assign_vec = vec1 - vec2;
        vec1 -= vec2;
        assert_eq!(vec1, sub_assign_vec);
    }
    #[test]
    fn test2_vector4sub_assign() {
        let mut vec1 = Vector4::new(1.5, -2., 4., 1.);
        let vec2 = Vector4::zero();
        let sub_assign_vec = vec1;
        vec1 -= vec2;
        assert_eq!(vec1, sub_assign_vec);
    }
    #[test]
    fn test3_vector4sub_assign() {
        let mut vec = Vector4::new(1., -3., 0., 5.);
        vec -= vec;
        assert_eq!(vec, Vector4::zero());
    }
    // Impl Mul
    #[test]
    fn test1_vector4mul() {
        let vec1 = Vector4::new(1., 4., 2., 0.);
        let vec2 = Vector4::new(-7., 3., -1., 169.);
        let exact_mul = 3.;
        assert_eq!(vec1 * vec2, vec2 * vec1);
        assert_eq!(vec1 * vec2, exact_mul);
    }
    #[test]
    fn test2_vector4mul() {
        let vec1 = Vector4::new(17.3, -5., 1.2, -7.);
        let vec2 = Vector4::zero();
        assert_eq!(vec1 * vec2, vec2 * vec1);
        assert_eq!(vec1 * vec2, 0.);
    }
    // Impl Mul<f32>
    #[test]
    fn test1_vector4mul_f32() {
        let vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 13.;
        let mul_vec = Vector4::new(13., -52., 1.3, -1.3);
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, mul_vec);
    }
    #[test]
    fn test2_vector4mul_f32() {
        let vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 2.;
        let mul_vec = vec + vec;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, mul_vec);
    }
    #[test]
    fn test3_vector4mul_f32() {
        let vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 1.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, vec);
    }
    #[test]
    fn test4_vector4mul_f32() {
        let vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 0.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, Vector4::zero());
    }
    #[test]
    fn test5_vector4mul_f32() {
        let vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = -1.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, -vec);
    }
    // Impl MulAssign<f32>
    #[test]
    fn test1_vector4mul_assign_f32() {
        let mut vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 13.;
        let mul_assign_vec = vec * value;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test2_vector4mul_assign_f32() {
        let mut vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 2.;
        let mul_assign_vec = vec + vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test3_vector4mul_assign_f32() {
        let mut vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 1.;
        let mul_assign_vec = vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test4_vector4mul_assign_f32() {
        let mut vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = 0.;
        vec *= value;
        assert_eq!(vec, Vector4::zero());
    }
    #[test]
    fn test5_vector4mul_assign_f32() {
        let mut vec = Vector4::new(1., -4., 0.1, -0.1);
        let value = -1.;
        let mul_assign_vec = -vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    // Impl Div<f32>
    #[test]
    fn test1_vector4div_f32() {
        let vec = Vector4::new(17.4, 9.3, -3., 1.);
        let value = 3.;
        let div_vec = Vector4::new(5.8, 3.1, -1., 0.3333333);
        assert_eq!(vec / value, div_vec)
    }
    #[test]
    fn test2_vector4div_f32() {
        let vec = Vector4::new(17.4, 9.3, -3., 1.);
        let value = 1.;
        assert_eq!(vec / value, vec);
    }
    #[test]
    fn test3_vector4div_f32() {
        let vec = Vector4::new(17.4, 9.3, -3., 1.);
        let value = -1.;
        assert_eq!(vec / value, -vec);
    }
    // Impl DivAssign<f32>
    #[test]
    fn test1_vector4div_assign_f32() {
        let mut vec = Vector4::new(17.4, 9.3, -3., 1.);
        let value = 3.;
        let div_assign_vec = vec / value;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
    #[test]
    fn test2_vector4div_assign_f32() {
        let mut vec = Vector4::new(17.4, 9.3, -3., 1.);
        let value = 1.;
        let div_assign_vec = vec;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
    #[test]
    fn test3_vector4div_assign_f32() {
        let mut vec = Vector4::new(17.4, 9.3, -3., 1.);
        let value = -1.;
        let div_assign_vec = -vec;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
}
