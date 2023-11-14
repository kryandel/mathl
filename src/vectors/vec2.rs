use super::{
    super::functions::constants::{EPSILON, PRECISION},
    Vector,
};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct Vector2 {
    pub x: f32,
    pub y: f32,
}
impl Vector2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}
impl Vector for Vector2 {
    fn dim(self) -> usize {
        2
    }

    fn len(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}
impl From<f32> for Vector2 {
    fn from(value: f32) -> Self {
        Self::new(value, value)
    }
}
impl Display for Vector2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_len = [self.x, self.y]
            .iter()
            .map(|comp| (*comp as i32).to_string().len())
            .max()
            .unwrap();
        let space = " ".repeat(1 + max_len + 1 + PRECISION + 1);
        writeln!(f, "\n┌{space}┐").unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.x).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} |", self.y).unwrap();
        writeln!(f, "└{space}┘").unwrap();
        Ok(())
    }
}
impl Neg for Vector2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}
impl Add for Vector2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl AddAssign for Vector2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Vector2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl SubAssign for Vector2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Vector2 {
    type Output = f32;

    fn mul(self, rhs: Self) -> Self::Output {
        self.x * rhs.x + self.y * rhs.y
    }
}
impl Mul<f32> for Vector2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}
impl Mul<Vector2> for f32 {
    type Output = Vector2;

    fn mul(self, rhs: Vector2) -> Self::Output {
        rhs * self
    }
}
impl MulAssign<f32> for Vector2 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl Div<f32> for Vector2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Vector2 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}
impl PartialEq for Vector2 {
    fn eq(&self, other: &Self) -> bool {
        (*self - *other).len() < EPSILON
    }
}
// impl Iterator for Vector2 {
//     type Item;

//     fn next(&mut self) -> Option<Self::Item> {
//         match  {

//         }
//         todo!()
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::constants::PI;

    // Constructors
    #[test]
    fn test1_vector2new() {
        let vec = Vector2 { x: 1., y: 2. };
        let vec_new = Vector2::new(1., 2.);
        assert_eq!(vec, vec_new);
    }
    #[test]
    fn test1_vector2from() {
        let vec = Vector2 { x: 1., y: 1. };
        let vec_from = Vector2::from(1.);
        assert_eq!(vec, vec_from);
    }
    #[test]
    fn test1_vector2zero() {
        let vec = Vector2 { x: 0., y: 0. };
        let vec_from = Vector2::zero();
        assert_eq!(vec, vec_from);
    }
    // Impl Display
    #[test]
    fn test1_vector2display() {
        let vec = Vector2::from(PI);
        println!("{vec}");
    }
    // Impl PartialEq
    #[test]
    fn test1_vector2partial_eq() {
        let vec1 = Vector2::new(1., 7.);
        let vec2 = Vector2::new(1., 8.);
        assert_ne!(vec1, vec2);
    }
    #[test]
    fn test2_vector2partial_eq() {
        let vec1 = Vector2::new(2.3, 2.3);
        let vec2 = Vector2::new(2.7, 2.7);
        assert_ne!(vec1, vec2);
    }
    #[test]
    fn test3_vector2partial_eq() {
        let vec = Vector2::new(-1.5, 2.3);
        assert_eq!(vec, vec);
    }
    // Method dim()
    #[test]
    fn test1_vector2dim() {
        let dim = Vector2::default().dim();
        assert_eq!(dim, 2)
    }
    // Method len()
    #[test]
    fn test1_vector2len() {
        let vec = Vector2::new(3., 4.);
        let len = vec.len();
        let exact_len = 5.;
        assert!((len - exact_len).abs() < EPSILON);
    }
    #[test]
    fn test2_vector2len() {
        let vec = Vector2::from(0.);
        let len = vec.len();
        assert!(len.abs() < EPSILON);
    }
    #[test]
    fn test3_vector2len() {
        let vec = Vector2::new(868.94, -1588.46);
        let len = vec.len();
        let exact_len = 1810.59711;
        assert!((len - exact_len).abs() < EPSILON);
    }
    #[test]
    // Method normalize()
    fn test1_vector2normalize() {
        let vec = Vector2::from(3.);
        let vec_normalize = vec.normalize();
        let exact_vec_normalize = Vector2::from(1. / 2.0_f32.sqrt());
        assert_eq!(vec_normalize, exact_vec_normalize);
    }
    #[test]
    fn test2_vector2normalize() {
        let vec = Vector2::new(13., -7.5);
        let vec_normalize = vec.normalize();
        let exact_vec_normalize = Vector2::new(0.866186, -0.499722);
        assert_eq!(vec_normalize, exact_vec_normalize);
    }
    #[test]
    #[should_panic]
    fn test3_vector2normalize() {
        Vector2::zero().normalize();
    }
    // Method angle(vector)
    #[test]
    fn test1_vector2angle() {
        let vec1 = Vector2::new(1., 0.);
        let vec2 = Vector2::new(0., 1.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test2_vector2angle() {
        let vec1 = Vector2::new(1., 0.);
        let vec2 = Vector2::new(0., -1.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI / 2.;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test3_vector2angle() {
        let vec1 = Vector2::new(2., 2.);
        let vec2 = Vector2::new(3., 3.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        assert!(angle1 < EPSILON && angle2 < EPSILON);
    }
    #[test]
    fn test4_vector2angle() {
        let vec1 = Vector2::new(-3., 3.);
        let vec2 = Vector2::new(1., -1.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = PI;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    fn test5_vector2angle() {
        let vec1 = Vector2::new(1., 2.);
        let vec2 = Vector2::new(7., 8.);
        let angle1 = vec1.angle(vec2);
        let angle2 = vec2.angle(vec1);
        let exact_angle = 0.255182;
        assert!((angle1 - exact_angle).abs() < EPSILON);
        assert!((angle2 - exact_angle).abs() < EPSILON);
    }
    #[test]
    #[should_panic]
    fn test6_vector2angle() {
        Vector2::new(1., 1.).angle(Vector2::zero());
    }
    #[test]
    #[should_panic]
    fn test7_vector2angle() {
        Vector2::zero().angle(Vector2::new(1., 1.));
    }
    // Method is_orthogonal_to(vector)
    #[test]
    fn test1_vector2is_orthogonal_to() {
        let vec1 = Vector2::new(1., 0.);
        let vec2 = Vector2::new(0., -7.);
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test2_vector2is_orthogonal_to() {
        let vec1 = Vector2::new(6., -2.);
        let vec2 = Vector2::new(1., 3.);
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test3_vector2is_orthogonal_to() {
        let vec1 = Vector2::new(1., 4.);
        let vec2 = Vector2::zero();
        assert!(vec1.is_orthogonal_to(vec2));
        assert!(vec2.is_orthogonal_to(vec1));
    }
    #[test]
    fn test4_vector2is_orthogonal_to() {
        let vec = Vector2::zero();
        assert!(vec.is_orthogonal_to(vec));
    }
    #[test]
    fn test5_vector2is_orthogonal_to() {
        let vec1 = Vector2::new(1., 4.);
        let vec2 = Vector2::new(7., -4.);
        assert_ne!(vec1.is_orthogonal_to(vec2), true);
        assert_ne!(vec2.is_orthogonal_to(vec1), true);
    }
    // Impl Neg
    #[test]
    fn test1_vector2neg() {
        let vec = Vector2::new(-1., 3.);
        let neg_vec = Vector2::new(1., -3.);
        assert_eq!(-vec, neg_vec)
    }
    #[test]
    fn test2_vector2neg() {
        let vec = Vector2::zero();
        assert_eq!(-vec, vec);
    }
    #[test]
    fn test3_vector2neg() {
        let vec = Vector2::new(1., 7.);
        assert_eq!(--vec, vec);
    }
    // Impl Add
    #[test]
    fn test1_vector2add() {
        let vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::new(-7.8, 4.);
        let add_vec = Vector2::new(-6.3, 2.);
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, add_vec)
    }
    #[test]
    fn test2_vector2add() {
        let vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::zero();
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, vec1);
    }
    #[test]
    fn test3_vector2add() {
        let vec = Vector2::zero();
        assert_eq!(vec + vec, vec);
    }
    #[test]
    fn test4_vector2add() {
        let vec1 = Vector2::new(1., -3.);
        let vec2 = -vec1;
        assert_eq!(vec1 + vec2, vec2 + vec1);
        assert_eq!(vec1 + vec2, Vector2::zero());
    }
    // Impl AddAsiign
    #[test]
    fn test1_vector2add_assign() {
        let mut vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::new(-7.8, 4.);
        let add_assign_vec = vec1 + vec2;
        vec1 += vec2;
        assert_eq!(vec1, add_assign_vec);
    }
    #[test]
    fn test2_vector2add_assign() {
        let mut vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::zero();
        let add_assign_vec = vec1;
        vec1 += vec2;
        assert_eq!(vec1, add_assign_vec);
    }
    #[test]
    fn test3_vector2add_assign() {
        let mut vec = Vector2::zero();
        vec += vec;
        assert_eq!(vec, Vector2::zero());
    }
    #[test]
    fn test4_vector2add_assign() {
        let mut vec1 = Vector2::new(1., -3.);
        let vec2 = -vec1;
        vec1 += vec2;
        assert_eq!(vec1, Vector2::zero());
    }
    // Impl Sub
    #[test]
    fn test1_vector2sub() {
        let vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::new(-7.8, 4.);
        let sub_vec = Vector2::new(9.3, -6.);
        assert_eq!(vec1 - vec2, -(vec2 - vec1));
        assert_eq!(vec1 - vec2, sub_vec)
    }
    #[test]
    fn test2_vector2sub() {
        let vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::zero();
        assert_eq!(vec1 - vec2, -(vec2 - vec1));
        assert_eq!(vec1 - vec2, vec1);
    }
    #[test]
    fn test3_vector2sub() {
        let vec = Vector2::new(1., -3.);
        assert_eq!(vec - vec, vec - vec);
        assert_eq!(vec - vec, Vector2::zero());
    }
    // Impl SubAssign
    #[test]
    fn test1_vector2sub_assign() {
        let mut vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::new(-7.8, 4.);
        let sub_assign_vec = vec1 - vec2;
        vec1 -= vec2;
        assert_eq!(vec1, sub_assign_vec);
    }
    #[test]
    fn test2_vector2sub_assign() {
        let mut vec1 = Vector2::new(1.5, -2.);
        let vec2 = Vector2::zero();
        let sub_assign_vec = vec1;
        vec1 -= vec2;
        assert_eq!(vec1, sub_assign_vec);
    }
    #[test]
    fn test3_vector2sub_assign() {
        let mut vec = Vector2::new(1., -3.);
        vec -= vec;
        assert_eq!(vec, Vector2::zero());
    }
    // Impl Mul
    #[test]
    fn test1_vector2mul() {
        let vec1 = Vector2::new(1., 4.);
        let vec2 = Vector2::new(-7., 3.);
        let exact_mul = 5.;
        assert_eq!(vec1 * vec2, vec2 * vec1);
        assert_eq!(vec1 * vec2, exact_mul);
    }
    #[test]
    fn test2_vector2mul() {
        let vec1 = Vector2::new(17.3, -5.);
        let vec2 = Vector2::zero();
        assert_eq!(vec1 * vec2, vec2 * vec1);
        assert_eq!(vec1 * vec2, 0.);
    }
    // Impl Mul<f32>
    #[test]
    fn test1_vector2mul_f32() {
        let vec = Vector2::new(1., -4.);
        let value = 13.;
        let mul_vec = Vector2::new(13., -52.);
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, mul_vec);
    }
    #[test]
    fn test2_vector2mul_f32() {
        let vec = Vector2::new(1., -4.);
        let value = 2.;
        let mul_vec = vec + vec;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, mul_vec);
    }
    #[test]
    fn test3_vector2mul_f32() {
        let vec = Vector2::new(1., -4.);
        let value = 1.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, vec);
    }
    #[test]
    fn test4_vector2mul_f32() {
        let vec = Vector2::new(1., -4.);
        let value = 0.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, Vector2::zero());
    }
    #[test]
    fn test5_vector2mul_f32() {
        let vec = Vector2::new(1., -4.);
        let value = -1.;
        assert_eq!(vec * value, value * vec);
        assert_eq!(vec * value, -vec);
    }
    // Impl MulAssign<f32>
    #[test]
    fn test1_vector2mul_assign_f32() {
        let mut vec = Vector2::new(1., -4.);
        let value = 13.;
        let mul_assign_vec = vec * value;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test2_vector2mul_assign_f32() {
        let mut vec = Vector2::new(1., -4.);
        let value = 2.;
        let mul_assign_vec = vec + vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test3_vector2mul_assign_f32() {
        let mut vec = Vector2::new(1., -4.);
        let value = 1.;
        let mul_assign_vec = vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    #[test]
    fn test4_vector2mul_assign_f32() {
        let mut vec = Vector2::new(1., -4.);
        let value = 0.;
        vec *= value;
        assert_eq!(vec, Vector2::zero());
    }
    #[test]
    fn test5_vector2mul_assign_f32() {
        let mut vec = Vector2::new(1., -4.);
        let value = -1.;
        let mul_assign_vec = -vec;
        vec *= value;
        assert_eq!(vec, mul_assign_vec);
    }
    // Impl Div<f32>
    #[test]
    fn test1_vector2div_f32() {
        let vec = Vector2::new(17.4, 9.3);
        let value = 3.;
        let div_vec = Vector2::new(5.8, 3.1);
        assert_eq!(vec / value, div_vec)
    }
    #[test]
    fn test2_vector2div_f32() {
        let vec = Vector2::new(17.4, 9.3);
        let value = 1.;
        assert_eq!(vec / value, vec);
    }
    #[test]
    fn test3_vector2div_f32() {
        let vec = Vector2::new(17.4, 9.3);
        let value = -1.;
        assert_eq!(vec / value, -vec);
    }
    // Impl DivAssign<f32>
    #[test]
    fn test1_vector2div_assign_f32() {
        let mut vec = Vector2::new(17.4, 9.3);
        let value = 3.;
        let div_assign_vec = vec / value;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
    #[test]
    fn test2_vector2div_assign_f32() {
        let mut vec = Vector2::new(17.4, 9.3);
        let value = 1.;
        let div_assign_vec = vec;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
    #[test]
    fn test3_vector2div_assign_f32() {
        let mut vec = Vector2::new(17.4, 9.3);
        let value = -1.;
        let div_assign_vec = -vec;
        vec /= value;
        assert_eq!(vec, div_assign_vec);
    }
}
