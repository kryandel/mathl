use super::Matrix;
use crate::{
    functions::{
        constants::{EPSILON, PRECISION},
        unpack,
    },
    Vector2,
};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct Matrix2 {
    data: [[f32; 2]; 2],
}
impl Matrix2 {
    pub fn new(data: [[f32; 2]; 2]) -> Self {
        Self { data }
    }

    pub fn from_rows(vec1: Vector2, vec2: Vector2) -> Self {
        let Vector2 { x: x1, y: y1 } = vec1;
        let Vector2 { x: x2, y: y2 } = vec2;
        Self::new([[x1, y1], [x2, y2]])
    }

    pub fn from_cols(vec1: Vector2, vec2: Vector2) -> Self {
        let Vector2 { x: x1, y: y1 } = vec1;
        let Vector2 { x: x2, y: y2 } = vec2;
        Self::new([[x1, x2], [y1, y2]])
    }
}
impl Matrix for Matrix2 {
    type Row = [f32; 2];
    type Column = [f32; 2];

    fn scalar(value: f32) -> Self {
        Self::new([[value, 0.], [0., value]])
    }

    fn dim(self) -> usize {
        2
    }
    fn det(self) -> f32 {
        self[(0, 0)] * self[(1, 1)] - self[(0, 1)] * self[(1, 0)]
    }

    fn get_row(self, i: usize) -> Self::Row {
        self.data[i]
    }
    fn get_col(self, j: usize) -> Self::Column {
        [self[(0, j)], self[(1, j)]]
    }
    fn transpose(self) -> Self {
        Self::new([[self[(0, 0)], self[(1, 0)]], [self[(0, 1)], self[(1, 1)]]])
    }

    fn try_invert(self) -> Option<Self>
    where
        Self: Sized,
    {
        let det = self.det();
        if det.abs() < EPSILON {
            return None;
        }
        Some(Self::new([[self[(1, 1)], -self[(0, 1)]], [-self[(1, 0)], self[(0, 0)]]]) / det)
    }
}
impl From<f32> for Matrix2 {
    fn from(value: f32) -> Self {
        Self::new([[value; 2]; 2])
    }
}
impl Index<(usize, usize)> for Matrix2 {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i][j]
    }
}
impl IndexMut<(usize, usize)> for Matrix2 {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[i][j]
    }
}
impl Display for Matrix2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_len = unpack(self.data)
            .iter()
            .map(|comp| (*comp as i32).to_string().len())
            .max()
            .unwrap();
        let space = " ".repeat(1 + (max_len + 1 + PRECISION + 1) * self.dim());
        writeln!(f, "\n┌{space}┐").unwrap();
        writeln!(
            f,
            "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |",
            self[(0, 0)],
            self[(0, 1)]
        )
        .unwrap();
        writeln!(
            f,
            "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |",
            self[(1, 0)],
            self[(1, 1)]
        )
        .unwrap();
        writeln!(f, "└{space}┘").unwrap();
        Ok(())
    }
}
impl Neg for Matrix2 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}
impl Add for Matrix2 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new([
            [self[(0, 0)] + rhs[(0, 0)], self[(0, 1)] + rhs[(0, 1)]],
            [self[(1, 0)] + rhs[(1, 0)], self[(1, 1)] + rhs[(1, 1)]],
        ])
    }
}
impl AddAssign for Matrix2 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs
    }
}
impl Sub for Matrix2 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl SubAssign for Matrix2 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Matrix2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Matrix2::default();
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    res[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        res
    }
}
impl MulAssign for Matrix2 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Mul<f32> for Matrix2 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new([
            [self[(0, 0)] * rhs, self[(0, 1)] * rhs],
            [self[(1, 0)] * rhs, self[(1, 1)] * rhs],
        ])
    }
}
impl Mul<Matrix2> for f32 {
    type Output = Matrix2;

    fn mul(self, rhs: Matrix2) -> Self::Output {
        rhs * self
    }
}
impl MulAssign<f32> for Matrix2 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl Div<f32> for Matrix2 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Matrix2 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs
    }
}
impl PartialEq for Matrix2 {
    fn eq(&self, other: &Self) -> bool {
        let matrix: Matrix2 = *self - *other;
        let mut sum = 0.;
        for i in 0..2 {
            for j in 0..2 {
                sum += matrix[(i, j)] * matrix[(i, j)];
            }
        }
        sum.sqrt() < EPSILON
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::functions::constants::PI;

    // Constructors
    #[test]
    fn test1_matrix2new() {
        let mat = Matrix2 {
            data: [[1., 2.], [0., -7.]],
        };
        let mat_new = Matrix2::new([[1., 2.], [0., -7.]]);
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix2from() {
        let mat = Matrix2 {
            data: [[2., 2.], [2., 2.]],
        };
        let mat_new = Matrix2::from(2.);
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix2idenity() {
        let mat = Matrix2 {
            data: [[1., 0.], [0., 1.]],
        };
        let mat_new = Matrix2::idenity();
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix2zero() {
        let mat = Matrix2 {
            data: [[0., 0.], [0., 0.]],
        };
        let mat_new = Matrix2::zero();
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix2scalar() {
        let mat = Matrix2 {
            data: [[-3.5, 0.], [0., -3.5]],
        };
        let mat_new = Matrix2::scalar(-3.5);
        assert_eq!(mat, mat_new);
    }
    // Impl Display
    #[test]
    fn test1_matrix2display() {
        let mat = Matrix2::scalar(PI);
        println!("{mat}");
    }
    // Impl PartialEq
    #[test]
    fn test1_matrix2partial_eq() {
        let mat1 = Matrix2::new([[1., 7.], [1., 7.]]);
        let mat2 = Matrix2::new([[1., 8.], [1., 8.]]);
        assert_ne!(mat1, mat2);
    }
    #[test]
    fn test2_matrix2partial_eq() {
        let mat1 = Matrix2::new([[2.3, 2.3], [2.3, 2.3]]);
        let mat2 = Matrix2::new([[2.7, 2.7], [2.7, 2.7]]);
        assert_ne!(mat1, mat2);
    }
    #[test]
    fn test3_matrix2partial_eq() {
        let mat = Matrix2::new([[-1.5, 2.3], [-1.1, 0.]]);
        assert_eq!(mat, mat);
    }
    // Method dim()
    #[test]
    fn test1_matrix2dim() {
        let dim = Matrix2::default().dim();
        assert_eq!(dim, 2)
    }
    // Getters
    #[test]
    fn test1_matrix2get_row() {
        let mat = Matrix2::new([[17., -7.], [2., 4.]]);
        let row = mat.get_row(0);
        assert_eq!(row, mat.data[0]);
    }
    #[test]
    fn test2_matrix2get_row() {
        let mat1 = Matrix2::new([[17., -7.], [2., 4.]]);
        let mat2 = Matrix2::new([[2., 4.], [-1., 11.]]);
        let (row1, row2) = (mat1.get_row(1), mat2.get_row(0));
        assert_eq!(row1, row2);
    }
    #[test]
    fn test1_matrix2get_col() {
        let mat = Matrix2::new([[17., -7.], [2., 4.]]);
        let col = mat.get_col(0);
        assert_eq!(col, [17., 2.]);
    }
    #[test]
    fn test2_matrix2get_col() {
        let mat1 = Matrix2::new([[17., -7.], [2., 4.]]);
        let mat2 = Matrix2::new([[2., 17.], [-1., 2.]]);
        let (col1, col2) = (mat1.get_col(0), mat2.get_col(1));
        assert_eq!(col1, col2);
    }
    // Method det()
    #[test]
    fn test1_matrix2det() {
        let mat = Matrix2::idenity();
        let exact_det = 1.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test2_matrix2det() {
        let mat = Matrix2::scalar(PI);
        let exact_det = PI * PI;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test3_matrix2det() {
        let mat = Matrix2::from(PI);
        let exact_det = 0.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test4_matrix2det() {
        let mat = Matrix2::new([[2., 1.], [7., 5.]]);
        let exact_det = 3.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    // Method transpose()
    #[test]
    fn test1_matrix2transpose() {
        let mat = Matrix2::new([[1., 7.], [-2., 1.]]);
        let exact_transpose = Matrix2::new([mat.get_col(0), mat.get_col(1)]);
        assert_eq!(mat.transpose(), exact_transpose);
    }
    #[test]
    fn test2_matrix2transpose() {
        let mat = Matrix2::new([[1., 7.], [7., 1.]]);
        assert_eq!(mat.transpose(), mat);
    }
    // Method invert()
    #[test]
    fn test1_matrix2invert() {
        let mat = Matrix2::new([[1., 7.], [-3., 4.]]);
        let exact_invert = Matrix2::new([[0.16, -0.28], [0.12, 0.04]]);
        assert_eq!(mat.invert(), exact_invert);
    }
    #[test]
    fn test2_matrix2invert() {
        let mat = Matrix2::idenity();
        assert_eq!(mat.invert(), mat);
    }
    #[test]
    #[should_panic]
    fn test3_matrix2invert() {
        Matrix2::new([[3., 6.], [7., 14.]]).invert();
    }
    // Impl Index<(usize, usize)>
    #[test]
    fn test1_matrix2index_usize_usize() {
        let mat = Matrix2::new([[1., PI], [-1., 0.]]);
        assert_eq!(mat[(0, 0)], 1.);
        assert_eq!(mat[(0, 1)], PI);
        assert_eq!(mat[(1, 0)], -1.);
        assert_eq!(mat[(1, 1)], 0.);
    }
    // Impl IndexMut<(usize, usize)>
    // #[test]
    // fn test1_matrix2indexmut_usize_usize() {
    //     let mut mat = Matrix2::new([[1., PI], [-1., 0.]]);
    //     for
    //     assert_eq!(mat[(0, 0)], 1.);
    //     assert_eq!(mat[(0, 1)], PI);
    //     assert_eq!(mat[(1, 0)], -1.);
    //     assert_eq!(mat[(1, 1)], 0.);
    // }
    // Impl Neg
    #[test]
    fn test1_matrix2neg() {
        let mat = Matrix2::new([[-1., 3.], [0., 9.]]);
        let neg_mat = Matrix2::new([[1., -3.], [0., -9.]]);
        assert_eq!(-mat, neg_mat)
    }
    #[test]
    fn test2_matrix2neg() {
        let mat = Matrix2::zero();
        assert_eq!(-mat, mat);
    }
    #[test]
    fn test3_matrix2neg() {
        let mat = Matrix2::new([[-1., 3.], [0., 9.]]);
        assert_eq!(--mat, mat);
    }
    // Impl Add
    #[test]
    fn test1_matrix2add() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::new([[-7.8, 4.], [0., 1.]]);
        let add_mat = Matrix2::new([[-6.3, 2.], [0., 4.1]]);
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, add_mat)
    }
    #[test]
    fn test2_matrix2add() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::zero();
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, mat1);
    }
    #[test]
    fn test3_matrix2add() {
        let mat = Matrix2::zero();
        assert_eq!(mat + mat, mat);
    }
    #[test]
    fn test4_matrix2add() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = -mat1;
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, Matrix2::zero());
    }
    // Impl AddAsiign
    #[test]
    fn test1_matrix2add_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::new([[-7.8, 4.], [0., 1.]]);
        let add_assign_mat = mat1 + mat2;
        mat1 += mat2;
        assert_eq!(mat1, add_assign_mat);
    }
    #[test]
    fn test2_matrix2add_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::zero();
        let add_assign_mat = mat1;
        mat1 += mat2;
        assert_eq!(mat1, add_assign_mat);
    }
    #[test]
    fn test3_matrix2add_assign() {
        let mut mat = Matrix2::zero();
        mat += mat;
        assert_eq!(mat, Matrix2::zero());
    }
    #[test]
    fn test4_matrix2add_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = -mat1;
        mat1 += mat2;
        assert_eq!(mat1, Matrix2::zero());
    }
    // Impl Sub
    #[test]
    fn test1_matrix2sub() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::new([[-7.8, 4.], [0., 1.]]);
        let sub_mat = Matrix2::new([[9.3, -6.], [0., 2.1]]);
        assert_eq!(mat1 - mat2, -(mat2 - mat1));
        assert_eq!(mat1 - mat2, sub_mat)
    }
    #[test]
    fn test2_matrix2sub() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::zero();
        assert_eq!(mat1 - mat2, -(mat2 - mat1));
        assert_eq!(mat1 - mat2, mat1);
    }
    #[test]
    fn test3_matrix2sub() {
        let mat = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        assert_eq!(mat - mat, mat - mat);
        assert_eq!(mat - mat, Matrix2::zero());
    }
    // Impl SubAssign
    #[test]
    fn test1_matrix2sub_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::new([[-7.8, 4.], [0., 1.]]);
        let sub_assign_mat = mat1 - mat2;
        mat1 -= mat2;
        assert_eq!(mat1, sub_assign_mat);
    }
    #[test]
    fn test2_matrix2sub_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::zero();
        let sub_assign_mat = mat1;
        mat1 -= mat2;
        assert_eq!(mat1, sub_assign_mat);
    }
    #[test]
    fn test3_matrix2sub_assign() {
        let mut mat = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        mat -= mat;
        assert_eq!(mat, Matrix2::zero());
    }
    // Impl Mul
    #[test]
    fn test1_matrix2mul() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::new([[-7.8, 4.], [0., 1.]]);
        let mul_mat = Matrix2::new([[-11.7, 4.], [0., 3.1]]);
        assert_ne!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, mul_mat);
    }
    #[test]
    fn test2_matrix2mul() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::zero();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, Matrix2::zero());
    }
    #[test]
    fn test3_matrix2mul() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::idenity();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, mat1);
    }
    #[test]
    fn test4_matrix2mul() {
        let mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = mat1.invert();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, Matrix2::idenity());
    }
    #[test]
    fn test5_matrix2mul() {
        let mat = Matrix2::idenity();
        assert_eq!(mat * mat, mat);
    }
    // Impl MulAssign
    #[test]
    fn test1_matrix2mul_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::new([[-7.8, 4.], [0., 1.]]);
        let mul_assign_mat = mat1 * mat2;
        mat1 *= mat2;
        assert_eq!(mat1, mul_assign_mat);
    }
    #[test]
    fn test2_matrix2mul_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::zero();
        mat1 *= mat2;
        assert_eq!(mat1, Matrix2::zero());
    }
    #[test]
    fn test3_matrix2mul_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = Matrix2::idenity();
        let mul_assign_mat = mat1;
        mat1 *= mat2;
        assert_eq!(mat1, mul_assign_mat);
    }
    #[test]
    fn test4_matrix2mul_assign() {
        let mut mat1 = Matrix2::new([[1.5, -2.], [0., 3.1]]);
        let mat2 = mat1.invert();
        mat1 *= mat2;
        assert_eq!(mat1, Matrix2::idenity());
    }
    #[test]
    fn test5_matrix2mul_assign() {
        let mut mat = Matrix2::idenity();
        mat *= mat;
        assert_eq!(mat, Matrix2::idenity());
    }
    // Impl Mul<f32>
    #[test]
    fn test1_matrix2mul_f32() {
        let mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 13.;
        let mul_mat = Matrix2::new([[13., -52.], [0., 1.3]]);
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mul_mat);
    }
    #[test]
    fn test2_matrix2mul_f32() {
        let mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 2.;
        let mul_mat = mat + mat;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mul_mat);
    }
    #[test]
    fn test3_matrix2mul_f32() {
        let mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 1.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mat);
    }
    #[test]
    fn test4_matrix2mul_f32() {
        let mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 0.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, Matrix2::zero());
    }
    #[test]
    fn test5_matrix2mul_f32() {
        let mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = -1.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, -mat);
    }
    // Impl MulAssign<f32>
    #[test]
    fn test1_matrix2mul_assign_f32() {
        let mut mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 13.;
        let mul_assign_mat = mat * value;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test2_matrix2mul_assign_f32() {
        let mut mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 2.;
        let mul_assign_mat = mat + mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test3_matrix2mul_assign_f32() {
        let mut mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 1.;
        let mul_assign_mat = mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test4_matrix2mul_assign_f32() {
        let mut mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = 0.;
        mat *= value;
        assert_eq!(mat, Matrix2::zero());
    }
    #[test]
    fn test5_matrix2mul_assign_f32() {
        let mut mat = Matrix2::new([[1., -4.], [0., 0.1]]);
        let value = -1.;
        let mul_assign_mat = -mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    // Impl Div<f32>
    #[test]
    fn test1_matrix2div_f32() {
        let mat = Matrix2::new([[17.4, 9.3], [3., 0.]]);
        let value = 3.;
        let div_mat = Matrix2::new([[5.8, 3.1], [1., 0.]]);
        assert_eq!(mat / value, div_mat)
    }
    #[test]
    fn test2_matrix2div_f32() {
        let mat = Matrix2::new([[17.4, 9.3], [3., 0.]]);
        let value = 1.;
        assert_eq!(mat / value, mat);
    }
    #[test]
    fn test3_matrix2div_f32() {
        let mat = Matrix2::new([[17.4, 9.3], [3., 0.]]);
        let value = -1.;
        assert_eq!(mat / value, -mat);
    }
    // Impl DivAssign<f32>
    #[test]
    fn test1_matrix2div_assign_f32() {
        let mut mat = Matrix2::new([[17.4, 9.3], [3., 0.]]);
        let value = 3.;
        let div_assign_mat = mat / value;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
    #[test]
    fn test2_matrix2div_assign_f32() {
        let mut mat = Matrix2::new([[17.4, 9.3], [3., 0.]]);
        let value = 1.;
        let div_assign_mat = mat;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
    #[test]
    fn test3_matrix2div_assign_f32() {
        let mut mat = Matrix2::new([[17.4, 9.3], [3., 0.]]);
        let value = -1.;
        let div_assign_mat = -mat;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
}
