use super::{mat2::Matrix2, Matrix};
use crate::{
    functions::{
        constants::{EPSILON, PRECISION},
        unpack,
    },
    Vector3,
};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct Matrix3 {
    data: [[f32; 3]; 3],
}
impl Matrix3 {
    pub fn new(data: [[f32; 3]; 3]) -> Self {
        Self { data }
    }

    pub fn from_rows(vec1: Vector3, vec2: Vector3, vec3: Vector3) -> Self {
        let Vector3 {
            x: x1,
            y: y1,
            z: z1,
        } = vec1;
        let Vector3 {
            x: x2,
            y: y2,
            z: z2,
        } = vec2;
        let Vector3 {
            x: x3,
            y: y3,
            z: z3,
        } = vec3;
        Self::new([[x1, y1, z1], [x2, y2, z2], [x3, y3, z3]])
    }

    pub fn from_cols(vec1: Vector3, vec2: Vector3, vec3: Vector3) -> Self {
        let Vector3 {
            x: x1,
            y: y1,
            z: z1,
        } = vec1;
        let Vector3 {
            x: x2,
            y: y2,
            z: z2,
        } = vec2;
        let Vector3 {
            x: x3,
            y: y3,
            z: z3,
        } = vec3;
        Self::new([[x1, x2, x3], [y1, y2, y3], [z1, z2, z3]])
    }
}
impl Matrix for Matrix3 {
    type Row = [f32; 3];
    type Column = [f32; 3];

    fn scalar(value: f32) -> Self {
        Self::new([[value, 0., 0.], [0., value, 0.], [0., 0., value]])
    }

    fn dim(self) -> usize {
        3
    }
    fn det(self) -> f32 {
        self[(0, 0)] * (self[(1, 1)] * self[(2, 2)] - self[(1, 2)] * self[(2, 1)])
            - self[(0, 1)] * (self[(1, 0)] * self[(2, 2)] - self[(1, 2)] * self[(2, 0)])
            + self[(0, 2)] * (self[(1, 0)] * self[(2, 1)] - self[(1, 1)] * self[(2, 0)])
    }

    fn get_row(self, i: usize) -> Self::Row {
        self.data[i]
    }
    fn get_col(self, j: usize) -> Self::Column {
        [self[(0, j)], self[(1, j)], self[(2, j)]]
    }

    fn transpose(self) -> Self {
        Self::new([
            [self[(0, 0)], self[(1, 0)], self[(2, 0)]],
            [self[(0, 1)], self[(1, 1)], self[(2, 1)]],
            [self[(0, 2)], self[(1, 2)], self[(2, 2)]],
        ])
    }
    fn try_invert(self) -> Option<Self>
    where
        Self: Sized,
    {
        let det = self.det();
        if det.abs() < EPSILON {
            return None;
        }
        let a00 = Matrix2::new([[self[(1, 1)], self[(1, 2)]], [self[(2, 1)], self[(2, 2)]]]).det();
        let a01 = Matrix2::new([[self[(1, 0)], self[(1, 2)]], [self[(2, 0)], self[(2, 2)]]]).det();
        let a02 = Matrix2::new([[self[(1, 0)], self[(1, 1)]], [self[(2, 0)], self[(2, 1)]]]).det();
        let a10 = Matrix2::new([[self[(0, 1)], self[(0, 2)]], [self[(2, 1)], self[(2, 2)]]]).det();
        let a11 = Matrix2::new([[self[(0, 0)], self[(0, 2)]], [self[(2, 0)], self[(2, 2)]]]).det();
        let a12 = Matrix2::new([[self[(0, 0)], self[(0, 1)]], [self[(2, 0)], self[(2, 1)]]]).det();
        let a20 = Matrix2::new([[self[(0, 1)], self[(0, 2)]], [self[(1, 1)], self[(1, 2)]]]).det();
        let a21 = Matrix2::new([[self[(0, 0)], self[(0, 2)]], [self[(1, 0)], self[(1, 2)]]]).det();
        let a22 = Matrix2::new([[self[(0, 0)], self[(0, 1)]], [self[(1, 0)], self[(1, 1)]]]).det();
        Some(Self::new([[a00, -a10, a20], [-a01, a11, -a21], [a02, -a12, a22]]) / det)
    }
}
impl From<f32> for Matrix3 {
    fn from(value: f32) -> Self {
        Self::new([[value; 3]; 3])
    }
}
impl Index<(usize, usize)> for Matrix3 {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i][j]
    }
}
impl IndexMut<(usize, usize)> for Matrix3 {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[i][j]
    }
}
impl Display for Matrix3 {
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
            "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |",
            self[(0, 0)],
            self[(0, 1)],
            self[(0, 2)]
        )
        .unwrap();
        writeln!(
            f,
            "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |",
            self[(1, 0)],
            self[(1, 1)],
            self[(1, 2)]
        )
        .unwrap();
        writeln!(
            f,
            "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |",
            self[(2, 0)],
            self[(2, 1)],
            self[(2, 2)]
        )
        .unwrap();
        writeln!(f, "└{space}┘").unwrap();
        Ok(())
    }
}
impl Neg for Matrix3 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}
impl Add for Matrix3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new([
            [
                self[(0, 0)] + rhs[(0, 0)],
                self[(0, 1)] + rhs[(0, 1)],
                self[(0, 2)] + rhs[(0, 2)],
            ],
            [
                self[(1, 0)] + rhs[(1, 0)],
                self[(1, 1)] + rhs[(1, 1)],
                self[(1, 2)] + rhs[(1, 2)],
            ],
            [
                self[(2, 0)] + rhs[(2, 0)],
                self[(2, 1)] + rhs[(2, 1)],
                self[(2, 2)] + rhs[(2, 2)],
            ],
        ])
    }
}
impl AddAssign for Matrix3 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Matrix3 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl SubAssign for Matrix3 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Matrix3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Matrix3::default();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    res[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        res
    }
}
impl MulAssign for Matrix3 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Mul<f32> for Matrix3 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new([
            [self[(0, 0)] * rhs, self[(0, 1)] * rhs, self[(0, 2)] * rhs],
            [self[(1, 0)] * rhs, self[(1, 1)] * rhs, self[(1, 2)] * rhs],
            [self[(2, 0)] * rhs, self[(2, 1)] * rhs, self[(2, 2)] * rhs],
        ])
    }
}
impl Mul<Matrix3> for f32 {
    type Output = Matrix3;

    fn mul(self, rhs: Matrix3) -> Self::Output {
        rhs * self
    }
}
impl MulAssign<f32> for Matrix3 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl Div<f32> for Matrix3 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Matrix3 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}
impl PartialEq for Matrix3 {
    fn eq(&self, other: &Self) -> bool {
        let matrix = *self - *other;
        let mut sum = 0.;
        for i in 0..3 {
            for j in 0..3 {
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
    fn test1_matrix3new() {
        let mat = Matrix3 {
            data: [[1., 2., 3.], [0., -7., 0.], [PI, 5., PI]],
        };
        let mat_new = Matrix3::new([[1., 2., 3.], [0., -7., 0.], [PI, 5., PI]]);
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix3from() {
        let mat = Matrix3 {
            data: [[2., 2., 2.], [2., 2., 2.], [2., 2., 2.]],
        };
        let mat_new = Matrix3::from(2.);
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix3idenity() {
        let mat = Matrix3 {
            data: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        };
        let mat_new = Matrix3::idenity();
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix3zero() {
        let mat = Matrix3 {
            data: [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]],
        };
        let mat_new = Matrix3::zero();
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix3scalar() {
        let mat = Matrix3 {
            data: [[-3.5, 0., 0.], [0., -3.5, 0.], [0., 0., -3.5]],
        };
        let mat_new = Matrix3::scalar(-3.5);
        assert_eq!(mat, mat_new);
    }
    // Impl Display
    #[test]
    fn test1_matrix3display() {
        let mat = Matrix3::scalar(PI);
        println!("{mat}");
    }
    // Impl PartialEq
    #[test]
    fn test1_matrix3partial_eq() {
        let mat1 = Matrix3::new([[1., 7., 1.], [1., 7., 1.], [1., 7., 1.]]);
        let mat2 = Matrix3::new([[1., 8., 1.], [1., 8., 1.], [1., 7., 1.]]);
        assert_ne!(mat1, mat2);
    }
    #[test]
    fn test2_matrix3partial_eq() {
        let mat1 = Matrix3::new([[2.3, 2.3, 2.3], [2.3, 2.3, 2.3], [2.3, 2.3, 2.3]]);
        let mat2 = Matrix3::new([[2.7, 2.7, 2.7], [2.7, 2.7, 2.7], [2.7, 2.7, 2.7]]);
        assert_ne!(mat1, mat2);
    }
    #[test]
    fn test3_matrix3partial_eq() {
        let mat = Matrix3::new([[-1.5, 2.3, 0.], [-1.1, 0., PI], [7.1, -13.25, 0.]]);
        assert_eq!(mat, mat);
    }
    // Method dim()
    #[test]
    fn test1_matrix3dim() {
        let dim = Matrix3::default().dim();
        assert_eq!(dim, 3)
    }
    // Getters
    #[test]
    fn test1_matrix3get_row() {
        let mat = Matrix3::new([[17., -7., PI], [2., 4., 1.], [3., -4., 5.]]);
        let row = mat.get_row(0);
        assert_eq!(row, mat.data[0]);
    }
    #[test]
    fn test2_matrix3get_row() {
        let mat1 = Matrix3::new([[17., -7., PI], [2., 4., 1.], [3., -4., 5.]]);
        let mat2 = Matrix3::new([[2., 4., 1.], [-1., 11., 17.], [9., -1., 0.]]);
        let (row1, row2) = (mat1.get_row(1), mat2.get_row(0));
        assert_eq!(row1, row2);
    }
    #[test]
    fn test1_matrix3get_col() {
        let mat = Matrix3::new([[17., -7., PI], [2., 4., 1.], [3., -4., 5.]]);
        let col = mat.get_col(0);
        assert_eq!(col, [17., 2., 3.]);
    }
    #[test]
    fn test2_matrix3get_col() {
        let mat1 = Matrix3::new([[17., -7., PI], [2., 4., 1.], [3., -4., 5.]]);
        let mat2 = Matrix3::new([[2., 17., 1.], [-1., 2., -3.], [0., 3., 19.]]);
        let (col1, col2) = (mat1.get_col(0), mat2.get_col(1));
        assert_eq!(col1, col2);
    }
    // Method det()
    #[test]
    fn test1_matrix3det() {
        let mat = Matrix3::idenity();
        let exact_det = 1.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test2_matrix3det() {
        let mat = Matrix3::scalar(PI);
        let exact_det = PI * PI * PI;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test3_matrix3det() {
        let mat = Matrix3::from(PI);
        let exact_det = 0.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test4_matrix3det() {
        let mat = Matrix3::new([[2., 1., 11.], [7., 5., 0.], [-3., 1., 0.]]);
        let exact_det = 242.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    // Method transpose()
    #[test]
    fn test1_matrix3transpose() {
        let mat = Matrix3::new([[1., 7., 0.], [-2., 1., PI], [0., -9., 4.]]);
        let exact_transpose = Matrix3::new([mat.get_col(0), mat.get_col(1), mat.get_col(2)]);
        assert_eq!(mat.transpose(), exact_transpose);
    }
    #[test]
    fn test2_matrix3transpose() {
        let mat = Matrix3::new([[1., 7., PI], [7., 1., -1.], [PI, -1., 2.]]);
        assert_eq!(mat.transpose(), mat);
    }
    // Method invert()
    #[test]
    fn test1_matrix3invert() {
        let mat = Matrix3::new([[1., 7., 0.], [-3., 4., 1.], [-8., -1., 0.]]);
        let exact_invert = Matrix3::new([
            [-0.0181818, 0., -0.127273],
            [0.145455, 0., 0.0181818],
            [-0.636363, 1., -0.454545],
        ]);
        assert_eq!(mat.invert(), exact_invert);
    }
    #[test]
    fn test2_matrix3invert() {
        let mat = Matrix3::idenity();
        assert_eq!(mat.invert(), mat);
    }
    #[test]
    #[should_panic]
    fn test3_matrix3invert() {
        Matrix3::new([[1., 2., 8.], [0., 2., 5.], [3., -4., -1.]]).invert();
    }
    // // Impl Index<(usize, usize)>
    //     #[test]
    //     fn test1_matrix3index_usize_usize() {
    //         let mat = Matrix3::new([[1., PI], [-1., 0.]]);
    //         assert_eq!(mat[(0, 0)], 1.);
    //         assert_eq!(mat[(0, 1)], PI);
    //         assert_eq!(mat[(1, 0)], -1.);
    //         assert_eq!(mat[(1, 1)], 0.);
    //     }
    // // Impl IndexMut<(usize, usize)>
    //     // #[test]
    //     // fn test1_matrix3indexmut_usize_usize() {
    //     //     let mut mat = Matrix3::new([[1., PI], [-1., 0.]]);
    //     //     for
    //     //     assert_eq!(mat[(0, 0)], 1.);
    //     //     assert_eq!(mat[(0, 1)], PI);
    //     //     assert_eq!(mat[(1, 0)], -1.);
    //     //     assert_eq!(mat[(1, 1)], 0.);
    //     // }
    // Impl Neg
    #[test]
    fn test1_matrix3neg() {
        let mat = Matrix3::new([[-1., 3., 2.], [0., 9., 17.], [3., 0., -1.]]);
        let neg_mat = Matrix3::new([[1., -3., -2.], [0., -9., -17.], [-3., 0., 1.]]);
        assert_eq!(-mat, neg_mat)
    }
    #[test]
    fn test2_matrix3neg() {
        let mat = Matrix3::zero();
        assert_eq!(-mat, mat);
    }
    #[test]
    fn test3_matrix3neg() {
        let mat = Matrix3::new([[-1., 3., 2.], [0., 9., 17.], [3., 0., -1.]]);
        assert_eq!(--mat, mat);
    }
    // Impl Add
    #[test]
    fn test1_matrix3add() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::new([[-7.8, 4., -1.], [0., 1., 1.2], [1.2, 1.1, 2.]]);
        let add_mat = Matrix3::new([[-6.3, 2., 0.], [0., 4.1, 3.7], [2.6, 1.1, 19.]]);
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, add_mat)
    }
    #[test]
    fn test2_matrix3add() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::zero();
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, mat1);
    }
    #[test]
    fn test3_matrix3add() {
        let mat = Matrix3::zero();
        assert_eq!(mat + mat, mat);
    }
    #[test]
    fn test4_matrix3add() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = -mat1;
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, Matrix3::zero());
    }
    // Impl AddAsiign
    #[test]
    fn test1_matrix3add_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::new([[-7.8, 4., -1.], [0., 1., 1.2], [1.2, 1.1, 2.]]);
        let add_assign_mat = mat1 + mat2;
        mat1 += mat2;
        assert_eq!(mat1, add_assign_mat);
    }
    #[test]
    fn test2_matrix3add_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::zero();
        let add_assign_mat = mat1;
        mat1 += mat2;
        assert_eq!(mat1, add_assign_mat);
    }
    #[test]
    fn test3_matrix3add_assign() {
        let mut mat = Matrix3::zero();
        mat += mat;
        assert_eq!(mat, Matrix3::zero());
    }
    #[test]
    fn test4_matrix3add_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = -mat1;
        mat1 += mat2;
        assert_eq!(mat1, Matrix3::zero());
    }
    // Impl Sub
    #[test]
    fn test1_matrix3sub() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::new([[-7.8, 4., -1.], [0., 1., 1.2], [1.2, 1.1, 2.]]);
        let sub_mat = Matrix3::new([[9.3, -6., 2.], [0., 2.1, 1.3], [0.2, -1.1, 15.]]);
        assert_eq!(mat1 - mat2, -(mat2 - mat1));
        assert!(mat1 - mat2 == sub_mat)
    }
    #[test]
    fn test2_matrix3sub() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::zero();
        assert_eq!(mat1 - mat2, -(mat2 - mat1));
        assert_eq!(mat1 - mat2, mat1);
    }
    #[test]
    fn test3_matrix3sub() {
        let mat = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        assert_eq!(mat - mat, mat - mat);
        assert_eq!(mat - mat, Matrix3::zero());
    }
    // Impl SubAssign
    #[test]
    fn test1_matrix3sub_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::new([[-7.8, 4., -1.], [0., 1., 1.2], [1.2, 1.1, 2.]]);
        let sub_assign_mat = mat1 - mat2;
        mat1 -= mat2;
        assert_eq!(mat1, sub_assign_mat);
    }
    #[test]
    fn test2_matrix3sub_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::zero();
        let sub_assign_mat = mat1;
        mat1 -= mat2;
        assert_eq!(mat1, sub_assign_mat);
    }
    #[test]
    fn test3_matrix3sub_assign() {
        let mut mat = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        mat -= mat;
        assert_eq!(mat, Matrix3::zero());
    }
    // Impl Mul
    #[test]
    fn test1_matrix3mul() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::new([[-7.8, 4., -1.], [0., 1., 1.2], [1.2, 1.1, 2.]]);
        let mul_mat = Matrix3::new([[-10.5, 5.1, -1.9], [3., 5.85, 8.72], [9.48, 24.3, 32.6]]);
        assert_ne!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, mul_mat);
    }
    #[test]
    fn test2_matrix3mul() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::zero();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, Matrix3::zero());
    }
    #[test]
    fn test3_matrix3mul() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::idenity();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, mat1);
    }
    #[test]
    fn test4_matrix3mul() {
        let mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = mat1.invert();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, Matrix3::idenity());
    }
    #[test]
    fn test5_matrix3mul() {
        let mat = Matrix3::idenity();
        assert_eq!(mat * mat, mat);
    }
    // Impl MulAssign
    #[test]
    fn test1_matrix3mul_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::new([[-7.8, 4., -1.], [0., 1., 1.2], [1.2, 1.1, 2.]]);
        let mul_assign_mat = mat1 * mat2;
        mat1 *= mat2;
        assert_eq!(mat1, mul_assign_mat);
    }
    #[test]
    fn test2_matrix3mul_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::zero();
        mat1 *= mat2;
        assert_eq!(mat1, Matrix3::zero());
    }
    #[test]
    fn test3_matrix3mul_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = Matrix3::idenity();
        let mul_assign_mat = mat1;
        mat1 *= mat2;
        assert_eq!(mat1, mul_assign_mat);
    }
    #[test]
    fn test4_matrix3mul_assign() {
        let mut mat1 = Matrix3::new([[1.5, -2., 1.], [0., 3.1, 2.5], [1.4, 0., 17.]]);
        let mat2 = mat1.invert();
        mat1 *= mat2;
        assert_eq!(mat1, Matrix3::idenity());
    }
    #[test]
    fn test5_matrix3mul_assign() {
        let mut mat = Matrix3::idenity();
        mat *= mat;
        assert_eq!(mat, Matrix3::idenity());
    }
    // Impl Mul<f32>
    #[test]
    fn test1_matrix3mul_f32() {
        let mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 13.;
        let mul_mat = Matrix3::new([[13., -52., -13.], [0., 1.3, 1.], [13. * PI, 26., 3.9]]);
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mul_mat);
    }
    #[test]
    fn test2_matrix3mul_f32() {
        let mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 2.;
        let mul_mat = mat + mat;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mul_mat);
    }
    #[test]
    fn test3_matrix3mul_f32() {
        let mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 1.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mat);
    }
    #[test]
    fn test4_matrix3mul_f32() {
        let mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 0.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, Matrix3::zero());
    }
    #[test]
    fn test5_matrix3mul_f32() {
        let mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = -1.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, -mat);
    }
    // Impl MulAssign<f32>
    #[test]
    fn test1_matrix3mul_assign_f32() {
        let mut mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 13.;
        let mul_assign_mat = mat * value;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test2_matrix3mul_assign_f32() {
        let mut mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 2.;
        let mul_assign_mat = mat + mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test3_matrix3mul_assign_f32() {
        let mut mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 1.;
        let mul_assign_mat = mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test4_matrix3mul_assign_f32() {
        let mut mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = 0.;
        mat *= value;
        assert_eq!(mat, Matrix3::zero());
    }
    #[test]
    fn test5_matrix3mul_assign_f32() {
        let mut mat = Matrix3::new([[1., -4., -1.], [0., 0.1, 1. / 13.], [PI, 2., 0.3]]);
        let value = -1.;
        let mul_assign_mat = -mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    // Impl Div<f32>
    #[test]
    fn test1_matrix3div_f32() {
        let mat = Matrix3::new([[17.4, 9.3, 1.], [3., 0., -12.], [3. * PI / 2., 6., 1.2]]);
        let value = 3.;
        let div_mat = Matrix3::new([[5.8, 3.1, 0.33333], [1., 0., -4.], [PI / 2., 2., 0.4]]);
        assert_eq!(mat / value, div_mat)
    }
    #[test]
    fn test2_matrix3div_f32() {
        let mat = Matrix3::new([[17.4, 9.3, 1.], [3., 0., -12.], [3. * PI / 2., 6., 1.2]]);
        let value = 1.;
        assert_eq!(mat / value, mat);
    }
    #[test]
    fn test3_matrix3div_f32() {
        let mat = Matrix3::new([[17.4, 9.3, 1.], [3., 0., -12.], [3. * PI / 2., 6., 1.2]]);
        let value = -1.;
        assert_eq!(mat / value, -mat);
    }
    // Impl DivAssign<f32>
    #[test]
    fn test1_matrix3div_assign_f32() {
        let mut mat = Matrix3::new([[17.4, 9.3, 1.], [3., 0., -12.], [3. * PI / 2., 6., 1.2]]);
        let value = 3.;
        let div_assign_mat = mat / value;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
    #[test]
    fn test2_matrix3div_assign_f32() {
        let mut mat = Matrix3::new([[17.4, 9.3, 1.], [3., 0., -12.], [3. * PI / 2., 6., 1.2]]);
        let value = 1.;
        let div_assign_mat = mat;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
    #[test]
    fn test3_matrix3div_assign_f32() {
        let mut mat = Matrix3::new([[17.4, 9.3, 1.], [3., 0., -12.], [3. * PI / 2., 6., 1.2]]);
        let value = -1.;
        let div_assign_mat = -mat;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
}
