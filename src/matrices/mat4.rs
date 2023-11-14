use super::{mat3::Matrix3, Matrix};
use crate::{
    functions::{
        constants::{EPSILON, PRECISION},
        unpack,
    },
    Vector4,
};
use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
};

#[derive(Default, Debug, Clone, Copy)]
pub struct Matrix4 {
    data: [[f32; 4]; 4],
}
impl Matrix4 {
    pub fn new(data: [[f32; 4]; 4]) -> Self {
        Self { data }
    }

    pub fn from_rows(vec1: Vector4, vec2: Vector4, vec3: Vector4, vec4: Vector4) -> Self {
        let Vector4 {
            x: x1,
            y: y1,
            z: z1,
            w: w1,
        } = vec1;
        let Vector4 {
            x: x2,
            y: y2,
            z: z2,
            w: w2,
        } = vec2;
        let Vector4 {
            x: x3,
            y: y3,
            z: z3,
            w: w3,
        } = vec3;
        let Vector4 {
            x: x4,
            y: y4,
            z: z4,
            w: w4,
        } = vec4;
        Self::new([
            [x1, y1, z1, w1],
            [x2, y2, z2, w2],
            [x3, y3, z3, w3],
            [x4, y4, z4, w4],
        ])
    }

    pub fn from_cols(vec1: Vector4, vec2: Vector4, vec3: Vector4, vec4: Vector4) -> Self {
        let Vector4 {
            x: x1,
            y: y1,
            z: z1,
            w: w1,
        } = vec1;
        let Vector4 {
            x: x2,
            y: y2,
            z: z2,
            w: w2,
        } = vec2;
        let Vector4 {
            x: x3,
            y: y3,
            z: z3,
            w: w3,
        } = vec3;
        let Vector4 {
            x: x4,
            y: y4,
            z: z4,
            w: w4,
        } = vec4;
        Self::new([
            [x1, x2, x3, x4],
            [y1, y2, y3, y4],
            [z1, z2, z3, z4],
            [w1, w2, w3, w4],
        ])
    }
}
impl Matrix for Matrix4 {
    type Row = [f32; 4];
    type Column = [f32; 4];

    fn scalar(value: f32) -> Self {
        Self::new([
            [value, 0., 0., 0.],
            [0., value, 0., 0.],
            [0., 0., value, 0.],
            [0., 0., 0., value],
        ])
    }

    fn dim(self) -> usize {
        4
    }
    fn det(self) -> f32 {
        self[(0, 0)]
            * Matrix3::new([
                [self[(1, 1)], self[(1, 2)], self[(1, 3)]],
                [self[(2, 1)], self[(2, 2)], self[(2, 3)]],
                [self[(3, 1)], self[(3, 2)], self[(3, 3)]],
            ])
            .det()
            - self[(0, 1)]
                * Matrix3::new([
                    [self[(1, 0)], self[(1, 2)], self[(1, 3)]],
                    [self[(2, 0)], self[(2, 2)], self[(2, 3)]],
                    [self[(3, 0)], self[(3, 2)], self[(3, 3)]],
                ])
                .det()
            + self[(0, 2)]
                * Matrix3::new([
                    [self[(1, 0)], self[(1, 1)], self[(1, 3)]],
                    [self[(2, 0)], self[(2, 1)], self[(2, 3)]],
                    [self[(3, 0)], self[(3, 1)], self[(3, 3)]],
                ])
                .det()
            - self[(0, 3)]
                * Matrix3::new([
                    [self[(1, 0)], self[(1, 1)], self[(1, 2)]],
                    [self[(2, 0)], self[(2, 1)], self[(2, 2)]],
                    [self[(3, 0)], self[(3, 1)], self[(3, 2)]],
                ])
                .det()
    }

    fn get_row(self, i: usize) -> Self::Row {
        self.data[i]
    }
    fn get_col(self, j: usize) -> Self::Column {
        [self[(0, j)], self[(1, j)], self[(2, j)], self[(3, j)]]
    }

    fn transpose(self) -> Self {
        Self::new([
            [self[(0, 0)], self[(1, 0)], self[(2, 0)], self[(3, 0)]],
            [self[(0, 1)], self[(1, 1)], self[(2, 1)], self[(3, 1)]],
            [self[(0, 2)], self[(1, 2)], self[(2, 2)], self[(3, 2)]],
            [self[(0, 3)], self[(1, 3)], self[(2, 3)], self[(3, 3)]],
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
        let a00 = Matrix3::new([
            [self[(1, 1)], self[(1, 2)], self[(1, 3)]],
            [self[(2, 1)], self[(2, 2)], self[(2, 3)]],
            [self[(3, 1)], self[(3, 2)], self[(3, 3)]],
        ])
        .det();
        let a01 = Matrix3::new([
            [self[(1, 0)], self[(1, 2)], self[(1, 3)]],
            [self[(2, 0)], self[(2, 2)], self[(2, 3)]],
            [self[(3, 0)], self[(3, 2)], self[(3, 3)]],
        ])
        .det();
        let a02 = Matrix3::new([
            [self[(1, 0)], self[(1, 1)], self[(1, 3)]],
            [self[(2, 0)], self[(2, 1)], self[(2, 3)]],
            [self[(3, 0)], self[(3, 1)], self[(3, 3)]],
        ])
        .det();
        let a03 = Matrix3::new([
            [self[(1, 0)], self[(1, 1)], self[(1, 2)]],
            [self[(2, 0)], self[(2, 1)], self[(2, 2)]],
            [self[(3, 0)], self[(3, 1)], self[(3, 2)]],
        ])
        .det();
        let a10 = Matrix3::new([
            [self[(0, 1)], self[(0, 2)], self[(0, 3)]],
            [self[(2, 1)], self[(2, 2)], self[(2, 3)]],
            [self[(3, 1)], self[(3, 2)], self[(3, 3)]],
        ])
        .det();
        let a11 = Matrix3::new([
            [self[(0, 0)], self[(0, 2)], self[(0, 3)]],
            [self[(2, 0)], self[(2, 2)], self[(2, 3)]],
            [self[(3, 0)], self[(3, 2)], self[(3, 3)]],
        ])
        .det();
        let a12 = Matrix3::new([
            [self[(0, 0)], self[(0, 1)], self[(0, 3)]],
            [self[(2, 0)], self[(2, 1)], self[(2, 3)]],
            [self[(3, 0)], self[(3, 1)], self[(3, 3)]],
        ])
        .det();
        let a13 = Matrix3::new([
            [self[(0, 0)], self[(0, 1)], self[(0, 2)]],
            [self[(2, 0)], self[(2, 1)], self[(2, 2)]],
            [self[(3, 0)], self[(3, 1)], self[(3, 2)]],
        ])
        .det();
        let a20 = Matrix3::new([
            [self[(0, 1)], self[(0, 2)], self[(0, 3)]],
            [self[(1, 1)], self[(1, 2)], self[(1, 3)]],
            [self[(3, 1)], self[(3, 2)], self[(3, 3)]],
        ])
        .det();
        let a21 = Matrix3::new([
            [self[(0, 0)], self[(0, 2)], self[(0, 3)]],
            [self[(1, 0)], self[(1, 2)], self[(1, 3)]],
            [self[(3, 0)], self[(3, 2)], self[(3, 3)]],
        ])
        .det();
        let a22 = Matrix3::new([
            [self[(0, 0)], self[(0, 1)], self[(0, 3)]],
            [self[(1, 0)], self[(1, 1)], self[(1, 3)]],
            [self[(3, 0)], self[(3, 1)], self[(3, 3)]],
        ])
        .det();
        let a23 = Matrix3::new([
            [self[(0, 0)], self[(0, 1)], self[(0, 2)]],
            [self[(1, 0)], self[(1, 1)], self[(1, 2)]],
            [self[(3, 0)], self[(3, 1)], self[(3, 2)]],
        ])
        .det();
        let a30 = Matrix3::new([
            [self[(0, 1)], self[(0, 2)], self[(0, 3)]],
            [self[(1, 1)], self[(1, 2)], self[(1, 3)]],
            [self[(2, 1)], self[(2, 2)], self[(2, 3)]],
        ])
        .det();
        let a31 = Matrix3::new([
            [self[(0, 0)], self[(0, 2)], self[(0, 3)]],
            [self[(1, 0)], self[(1, 2)], self[(1, 3)]],
            [self[(2, 0)], self[(2, 2)], self[(2, 3)]],
        ])
        .det();
        let a32 = Matrix3::new([
            [self[(0, 0)], self[(0, 1)], self[(0, 3)]],
            [self[(1, 0)], self[(1, 1)], self[(1, 3)]],
            [self[(2, 0)], self[(2, 1)], self[(2, 3)]],
        ])
        .det();
        let a33 = Matrix3::new([
            [self[(0, 0)], self[(0, 1)], self[(0, 2)]],
            [self[(1, 0)], self[(1, 1)], self[(1, 2)]],
            [self[(2, 0)], self[(2, 1)], self[(2, 2)]],
        ])
        .det();
        Some(
            Self::new([
                [a00, -a10, a20, -a30],
                [-a01, a11, -a21, a31],
                [a02, -a12, a22, -a32],
                [-a03, a13, -a23, a33],
            ]) / det,
        )
    }
}
impl From<f32> for Matrix4 {
    fn from(value: f32) -> Self {
        Self::new([[value; 4]; 4])
    }
}
impl Index<(usize, usize)> for Matrix4 {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        &self.data[i][j]
    }
}
impl IndexMut<(usize, usize)> for Matrix4 {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (i, j) = index;
        &mut self.data[i][j]
    }
}
impl Display for Matrix4 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let max_len = unpack(self.data)
            .iter()
            .map(|comp| (*comp as i32).to_string().len())
            .max()
            .unwrap();
        let space = " ".repeat(1 + (max_len + 1 + PRECISION + 1) * self.dim());
        writeln!(f, "\n┌{space}┐").unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |", self[(0, 0)], self[(0, 1)], self[(0, 2)], self[(0, 3)]).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |", self[(1, 0)], self[(1, 1)], self[(1, 2)], self[(1, 3)]).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |", self[(2, 0)], self[(2, 1)], self[(2, 2)], self[(2, 3)]).unwrap();
        writeln!(f, "| {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} {:>max_len$.PRECISION$} |", self[(3, 0)], self[(3, 1)], self[(3, 2)], self[(3, 3)]).unwrap();
        writeln!(f, "└{space}┘").unwrap();
        Ok(())
    }
}
impl Neg for Matrix4 {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self * -1.
    }
}
impl Add for Matrix4 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new([
            [
                self[(0, 0)] + rhs[(0, 0)],
                self[(0, 1)] + rhs[(0, 1)],
                self[(0, 2)] + rhs[(0, 2)],
                self[(0, 3)] + rhs[(0, 3)],
            ],
            [
                self[(1, 0)] + rhs[(1, 0)],
                self[(1, 1)] + rhs[(1, 1)],
                self[(1, 2)] + rhs[(1, 2)],
                self[(1, 3)] + rhs[(1, 3)],
            ],
            [
                self[(2, 0)] + rhs[(2, 0)],
                self[(2, 1)] + rhs[(2, 1)],
                self[(2, 2)] + rhs[(2, 2)],
                self[(2, 3)] + rhs[(2, 3)],
            ],
            [
                self[(3, 0)] + rhs[(3, 0)],
                self[(3, 1)] + rhs[(3, 1)],
                self[(3, 2)] + rhs[(3, 2)],
                self[(3, 3)] + rhs[(3, 3)],
            ],
        ])
    }
}
impl AddAssign for Matrix4 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl Sub for Matrix4 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self + -rhs
    }
}
impl SubAssign for Matrix4 {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl Mul for Matrix4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut res = Matrix4::default();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    res[(i, j)] += self[(i, k)] * rhs[(k, j)];
                }
            }
        }
        res
    }
}
impl MulAssign for Matrix4 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}
impl Mul<f32> for Matrix4 {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new([
            [
                self[(0, 0)] * rhs,
                self[(0, 1)] * rhs,
                self[(0, 2)] * rhs,
                self[(0, 3)] * rhs,
            ],
            [
                self[(1, 0)] * rhs,
                self[(1, 1)] * rhs,
                self[(1, 2)] * rhs,
                self[(1, 3)] * rhs,
            ],
            [
                self[(2, 0)] * rhs,
                self[(2, 1)] * rhs,
                self[(2, 2)] * rhs,
                self[(2, 3)] * rhs,
            ],
            [
                self[(3, 0)] * rhs,
                self[(3, 1)] * rhs,
                self[(3, 2)] * rhs,
                self[(3, 3)] * rhs,
            ],
        ])
    }
}
impl Mul<Matrix4> for f32 {
    type Output = Matrix4;

    fn mul(self, rhs: Matrix4) -> Self::Output {
        rhs * self
    }
}
impl MulAssign<f32> for Matrix4 {
    fn mul_assign(&mut self, rhs: f32) {
        *self = *self * rhs;
    }
}
impl Div<f32> for Matrix4 {
    type Output = Self;

    fn div(self, rhs: f32) -> Self::Output {
        self * (1. / rhs)
    }
}
impl DivAssign<f32> for Matrix4 {
    fn div_assign(&mut self, rhs: f32) {
        *self = *self / rhs;
    }
}
impl PartialEq for Matrix4 {
    fn eq(&self, other: &Self) -> bool {
        let matrix = *self - *other;
        let mut sum = 0.;
        for i in 0..4 {
            for j in 0..4 {
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
    fn test1_matrix4new() {
        let mat = Matrix4 {
            data: [
                [1., 2., 3., 4.],
                [0., -7., 0., 1.5],
                [PI, 5., PI, -0.1],
                [0., -7., 0., 1.5],
            ],
        };
        let mat_new = Matrix4::new([
            [1., 2., 3., 4.],
            [0., -7., 0., 1.5],
            [PI, 5., PI, -0.1],
            [0., -7., 0., 1.5],
        ]);
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix4from() {
        let mat = Matrix4 {
            data: [
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
                [2., 2., 2., 2.],
            ],
        };
        let mat_new = Matrix4::from(2.);
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix4idenity() {
        let mat = Matrix4 {
            data: [
                [1., 0., 0., 0.],
                [0., 1., 0., 0.],
                [0., 0., 1., 0.],
                [0., 0., 0., 1.],
            ],
        };
        let mat_new = Matrix4::idenity();
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix4zero() {
        let mat = Matrix4 {
            data: [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
        };
        let mat_new = Matrix4::zero();
        assert_eq!(mat, mat_new);
    }
    #[test]
    fn test1_matrix4scalar() {
        let mat = Matrix4 {
            data: [
                [-3.5, 0., 0., 0.],
                [0., -3.5, 0., 0.],
                [0., 0., -3.5, 0.],
                [0., 0., 0., -3.5],
            ],
        };
        let mat_new = Matrix4::scalar(-3.5);
        assert_eq!(mat, mat_new);
    }
    // Impl Display
    #[test]
    fn test1_matrix4display() {
        let mat = Matrix4::scalar(PI);
        println!("{mat}");
    }
    // Impl PartialEq
    #[test]
    fn test1_matrix4partial_eq() {
        let mat1 = Matrix4::new([
            [1., 7., 1., 0.],
            [1., 7., 1., 0.],
            [1., 7., 1., 0.],
            [1., 7., 1., 0.],
        ]);
        let mat2 = Matrix4::new([
            [1., 8., 1., 0.],
            [1., 8., 1., 0.],
            [1., 7., 1., 0.],
            [1., 7., 1., 0.],
        ]);
        assert_ne!(mat1, mat2);
    }
    #[test]
    fn test2_matrix4partial_eq() {
        let mat1 = Matrix4::new([
            [2.3, 2.3, 2.3, 2.3],
            [2.3, 2.3, 2.3, 2.3],
            [2.3, 2.3, 2.3, 2.3],
            [2.3, 2.3, 2.3, 2.3],
        ]);
        let mat2 = Matrix4::new([
            [2.7, 2.7, 2.7, 2.7],
            [2.7, 2.7, 2.7, 2.7],
            [2.7, 2.7, 2.7, 2.7],
            [2.7, 2.7, 2.7, 2.7],
        ]);
        assert_ne!(mat1, mat2);
    }
    #[test]
    fn test3_matrix4partial_eq() {
        let mat = Matrix4::new([
            [-1.5, 2.3, 0., PI / 2.],
            [-1.1, 0., PI, 1.7],
            [7.1, -13.25, 0., -2.],
            [1., 4.5, -1., -6.9],
        ]);
        assert_eq!(mat, mat);
    }
    // Method dim()
    #[test]
    fn test1_matrix4dim() {
        let dim = Matrix4::default().dim();
        assert_eq!(dim, 4)
    }
    // Getters
    #[test]
    fn test1_matrix4get_row() {
        let mat = Matrix4::new([
            [17., -7., PI, 11.],
            [2., 4., 1., 0.],
            [3., -4., 5., -1.],
            [2., 4., 1., 0.],
        ]);
        let row = mat.get_row(0);
        assert_eq!(row, mat.data[0]);
    }
    #[test]
    fn test2_matrix4get_row() {
        let mat1 = Matrix4::new([
            [17., -7., PI, 11.],
            [2., 4., 1., 0.],
            [3., -4., 5., -1.],
            [2., 4., 1., 0.],
        ]);
        let mat2 = Matrix4::new([
            [2., 4., 1., 0.],
            [-1., 11., 17., PI],
            [9., -1., 0., -9.],
            [1., 1., 0., 1.],
        ]);
        let (row1, row2) = (mat1.get_row(1), mat2.get_row(0));
        assert_eq!(row1, row2);
    }
    #[test]
    fn test1_matrix4get_col() {
        let mat = Matrix4::new([
            [17., -7., PI, 11.],
            [2., 4., 1., 0.],
            [3., -4., 5., -1.],
            [2., 4., 1., 0.],
        ]);
        let col = mat.get_col(0);
        assert_eq!(col, [17., 2., 3., 2.]);
    }
    #[test]
    fn test2_matrix4get_col() {
        let mat1 = Matrix4::new([
            [17., -7., PI, 11.],
            [2., 4., 1., 0.],
            [3., -4., 5., -1.],
            [2., 4., 1., 0.],
        ]);
        let mat2 = Matrix4::new([
            [2., 17., 1., 0.],
            [-1., 2., -3., PI],
            [0., 3., 19., 0.],
            [7.9, 2., 0., 6.9],
        ]);
        let (col1, col2) = (mat1.get_col(0), mat2.get_col(1));
        assert_eq!(col1, col2);
    }
    // Method det()
    #[test]
    fn test1_matrix4det() {
        let mat = Matrix4::idenity();
        let exact_det = 1.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test2_matrix4det() {
        let mat = Matrix4::scalar(PI);
        let exact_det = PI * PI * PI * PI;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test3_matrix4det() {
        let mat = Matrix4::from(PI);
        let exact_det = 0.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    #[test]
    fn test4_matrix4det() {
        let mat = Matrix4::new([
            [2., 1., 11., 0.],
            [7., 5., 0., -1.],
            [-3., 1., 0., 2.],
            [0., 2., 0., -1.],
        ]);
        let exact_det = -484.;
        assert!((mat.det() - exact_det).abs() < EPSILON);
    }
    // Method transpose()
    #[test]
    fn test1_matrix4transpose() {
        let mat = Matrix4::new([
            [1., 7., 0., 3.],
            [-2., 1., PI, -9.],
            [0., -9., 4., 1.3],
            [4.5, 0., -7., 11.],
        ]);
        let exact_transpose = Matrix4::new([
            mat.get_col(0),
            mat.get_col(1),
            mat.get_col(2),
            mat.get_col(3),
        ]);
        assert_eq!(mat.transpose(), exact_transpose);
    }
    #[test]
    fn test2_matrix4transpose() {
        let mat = Matrix4::new([
            [1., 7., PI, 0.],
            [7., 1., -1., 1.5],
            [PI, -1., 2., 1.5],
            [0., 1.5, 1.5, 7.],
        ]);
        assert_eq!(mat.transpose(), mat);
    }
    // Method invert()
    #[test]
    fn test1_matrix4invert() {
        let mat = Matrix4::new([
            [1., 7., 0., 7.],
            [-3., 4., 1., 0.],
            [-8., -1., 0., 11.],
            [1., 3., -3., 0.],
        ]);
        let exact_invert = Matrix4::new([
            [0.09839, -0.150268, -0.0626118, -0.0500894],
            [0.0524747, 0.119857, -0.033393, 0.0399532],
            [0.0852713, 0.0697674, -0.0542636, -0.310078],
            [0.0763268, -0.09839, 0.0423375, -0.0327967],
        ]);
        assert_eq!(mat.invert(), exact_invert);
    }
    #[test]
    fn test2_matrix4invert() {
        let mat = Matrix4::idenity();
        assert_eq!(mat.invert(), mat);
    }
    #[test]
    #[should_panic]
    fn test3_matrix4invert() {
        Matrix4::new([
            [0., 0., 128., -143.],
            [-3., 4., 1., 0.],
            [-8., -1., 0., 11.],
            [1., 3., -3., 0.],
        ])
        .invert();
    }
    // // // Impl Index<(usize, usize)>
    // //     #[test]
    // //     fn test1_matrix4index_usize_usize() {
    // //         let mat = Matrix4::new([[1., PI], [-1., 0.]]);
    // //         assert_eq!(mat[(0, 0)], 1.);
    // //         assert_eq!(mat[(0, 1)], PI);
    // //         assert_eq!(mat[(1, 0)], -1.);
    // //         assert_eq!(mat[(1, 1)], 0.);
    // //     }
    // // // Impl IndexMut<(usize, usize)>
    // //     // #[test]
    // //     // fn test1_matrix4indexmut_usize_usize() {
    // //     //     let mut mat = Matrix4::new([[1., PI], [-1., 0.]]);
    // //     //     for
    // //     //     assert_eq!(mat[(0, 0)], 1.);
    // //     //     assert_eq!(mat[(0, 1)], PI);
    // //     //     assert_eq!(mat[(1, 0)], -1.);
    // //     //     assert_eq!(mat[(1, 1)], 0.);
    // //     // }
    // Impl Neg
    #[test]
    fn test1_matrix4neg() {
        let mat = Matrix4::new([
            [-1., 3., 2., PI],
            [0., 9., 17., 1.],
            [3., 0., -1., 12.],
            [1., -3., 14., 5.],
        ]);
        let neg_mat = Matrix4::new([
            [1., -3., -2., -PI],
            [0., -9., -17., -1.],
            [-3., 0., 1., -12.],
            [-1., 3., -14., -5.],
        ]);
        assert_eq!(-mat, neg_mat)
    }
    #[test]
    fn test2_matrix4neg() {
        let mat = Matrix4::zero();
        assert_eq!(-mat, mat);
    }
    #[test]
    fn test3_matrix4neg() {
        let mat = Matrix4::new([
            [-1., 3., 2., PI],
            [0., 9., 17., 1.],
            [3., 0., -1., 12.],
            [1., -3., 14., 5.],
        ]);
        assert_eq!(--mat, mat);
    }
    // Impl Add
    #[test]
    fn test1_matrix4add() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::new([
            [-7.8, 4., -1., 4.],
            [0., 1., 1.2, 1.1],
            [1.2, 1.1, 2., 0.7],
            [1., 0., 1., 2.],
        ]);
        let add_mat = Matrix4::new([
            [-6.3, 2., 0., 6.],
            [0., 4.1, 3.7, 2.2],
            [2.6, 1.1, 19., 1.],
            [4.2, 0., 4., 3.],
        ]);
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, add_mat)
    }
    #[test]
    fn test2_matrix4add() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::zero();
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, mat1);
    }
    #[test]
    fn test3_matrix4add() {
        let mat = Matrix4::zero();
        assert_eq!(mat + mat, mat);
    }
    #[test]
    fn test4_matrix4add() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = -mat1;
        assert_eq!(mat1 + mat2, mat2 + mat1);
        assert_eq!(mat1 + mat2, Matrix4::zero());
    }
    // Impl AddAsiign
    #[test]
    fn test1_matrix4add_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::new([
            [-7.8, 4., -1., 4.],
            [0., 1., 1.2, 1.1],
            [1.2, 1.1, 2., 0.7],
            [1., 0., 1., 2.],
        ]);
        let add_assign_mat = mat1 + mat2;
        mat1 += mat2;
        assert_eq!(mat1, add_assign_mat);
    }
    #[test]
    fn test2_matrix4add_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::zero();
        let add_assign_mat = mat1;
        mat1 += mat2;
        assert_eq!(mat1, add_assign_mat);
    }
    #[test]
    fn test3_matrix4add_assign() {
        let mut mat = Matrix4::zero();
        mat += mat;
        assert_eq!(mat, Matrix4::zero());
    }
    #[test]
    fn test4_matrix4add_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = -mat1;
        mat1 += mat2;
        assert_eq!(mat1, Matrix4::zero());
    }
    // Impl Sub
    #[test]
    fn test1_matrix4sub() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::new([
            [-7.8, 4., -1., 4.],
            [0., 1., 1.2, 1.1],
            [1.2, 1.1, 2., 0.7],
            [1., 0., 1., 2.],
        ]);
        let sub_mat = Matrix4::new([
            [9.3, -6., 2., -2.],
            [0., 2.1, 1.3, 0.],
            [0.2, -1.1, 15., -0.4],
            [2.2, 0., 2., -1.],
        ]);
        assert_eq!(mat1 - mat2, -(mat2 - mat1));
        assert!(mat1 - mat2 == sub_mat)
    }
    #[test]
    fn test2_matrix4sub() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::zero();
        assert_eq!(mat1 - mat2, -(mat2 - mat1));
        assert_eq!(mat1 - mat2, mat1);
    }
    #[test]
    fn test3_matrix4sub() {
        let mat = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        assert_eq!(mat - mat, mat - mat);
        assert_eq!(mat - mat, Matrix4::zero());
    }
    // Impl SubAssign
    #[test]
    fn test1_matrix4sub_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::new([
            [-7.8, 4., -1., 4.],
            [0., 1., 1.2, 1.1],
            [1.2, 1.1, 2., 0.7],
            [1., 0., 1., 2.],
        ]);
        let sub_assign_mat = mat1 - mat2;
        mat1 -= mat2;
        assert_eq!(mat1, sub_assign_mat);
    }
    #[test]
    fn test2_matrix4sub_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::zero();
        let sub_assign_mat = mat1;
        mat1 -= mat2;
        assert_eq!(mat1, sub_assign_mat);
    }
    #[test]
    fn test3_matrix4sub_assign() {
        let mut mat = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        mat -= mat;
        assert_eq!(mat, Matrix4::zero());
    }
    // Impl Mul
    #[test]
    fn test1_matrix4mul() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::new([
            [-7.8, 4., -1., 4.],
            [0., 1., 1.2, 1.1],
            [1.2, 1.1, 2., 0.7],
            [1., 0., 1., 2.],
        ]);
        let mul_mat = Matrix4::new([
            [-8.5, 5.1, 0.1, 8.5],
            [4.1, 5.85, 9.82, 7.36],
            [9.78, 24.3, 32.9, 18.1],
            [-20.36, 16.1, 3.8, 16.9],
        ]);
        assert_ne!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, mul_mat);
    }
    #[test]
    fn test2_matrix4mul() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::zero();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, Matrix4::zero());
    }
    #[test]
    fn test3_matrix4mul() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::idenity();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, mat1);
    }
    #[test]
    fn test4_matrix4mul() {
        let mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = mat1.invert();
        assert_eq!(mat1 * mat2, mat2 * mat1);
        assert_eq!(mat1 * mat2, Matrix4::idenity());
    }
    #[test]
    fn test5_matrix4mul() {
        let mat = Matrix4::idenity();
        assert_eq!(mat * mat, mat);
    }
    // Impl MulAssign
    #[test]
    fn test1_matrix4mul_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::new([
            [-7.8, 4., -1., 4.],
            [0., 1., 1.2, 1.1],
            [1.2, 1.1, 2., 0.7],
            [1., 0., 1., 2.],
        ]);
        let mul_assign_mat = mat1 * mat2;
        mat1 *= mat2;
        assert_eq!(mat1, mul_assign_mat);
    }
    #[test]
    fn test2_matrix4mul_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::zero();
        mat1 *= mat2;
        assert_eq!(mat1, Matrix4::zero());
    }
    #[test]
    fn test3_matrix4mul_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = Matrix4::idenity();
        let mul_assign_mat = mat1;
        mat1 *= mat2;
        assert_eq!(mat1, mul_assign_mat);
    }
    #[test]
    fn test4_matrix4mul_assign() {
        let mut mat1 = Matrix4::new([
            [1.5, -2., 1., 2.],
            [0., 3.1, 2.5, 1.1],
            [1.4, 0., 17., 0.3],
            [3.2, 0., 3., 1.],
        ]);
        let mat2 = mat1.invert();
        mat1 *= mat2;
        assert_eq!(mat1, Matrix4::idenity());
    }
    #[test]
    fn test5_matrix4mul_assign() {
        let mut mat = Matrix4::idenity();
        mat *= mat;
        assert_eq!(mat, Matrix4::idenity());
    }
    // Impl Mul<f32>
    #[test]
    fn test1_matrix4mul_f32() {
        let mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 13.;
        let mul_mat = Matrix4::new([
            [13., -52., -13., 26.],
            [0., 1.3, 1., 0.],
            [13. * PI, 26., 3.9, 13.],
            [130., -13., 0., 39.],
        ]);
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mul_mat);
    }
    #[test]
    fn test2_matrix4mul_f32() {
        let mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 2.;
        let mul_mat = mat + mat;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mul_mat);
    }
    #[test]
    fn test3_matrix4mul_f32() {
        let mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 1.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, mat);
    }
    #[test]
    fn test4_matrix4mul_f32() {
        let mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 0.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, Matrix4::zero());
    }
    #[test]
    fn test5_matrix4mul_f32() {
        let mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = -1.;
        assert_eq!(mat * value, value * mat);
        assert_eq!(mat * value, -mat);
    }
    // Impl MulAssign<f32>
    #[test]
    fn test1_matrix4mul_assign_f32() {
        let mut mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 13.;
        let mul_assign_mat = mat * value;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test2_matrix4mul_assign_f32() {
        let mut mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 2.;
        let mul_assign_mat = mat + mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test3_matrix4mul_assign_f32() {
        let mut mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 1.;
        let mul_assign_mat = mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    #[test]
    fn test4_matrix4mul_assign_f32() {
        let mut mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = 0.;
        mat *= value;
        assert_eq!(mat, Matrix4::zero());
    }
    #[test]
    fn test5_matrix4mul_assign_f32() {
        let mut mat = Matrix4::new([
            [1., -4., -1., 2.],
            [0., 0.1, 1. / 13., 0.],
            [PI, 2., 0.3, 1.],
            [10., -1., 0., 3.],
        ]);
        let value = -1.;
        let mul_assign_mat = -mat;
        mat *= value;
        assert_eq!(mat, mul_assign_mat);
    }
    // Impl Div<f32>
    #[test]
    fn test1_matrix4div_f32() {
        let mat = Matrix4::new([
            [17.4, 9.3, 1., 0.],
            [3., 0., -12., 6.],
            [3. * PI / 2., 6., 1.2, 10.],
            [33., 9., 1.8, -3.],
        ]);
        let value = 3.;
        let div_mat = Matrix4::new([
            [5.8, 3.1, 0.33333, 0.],
            [1., 0., -4., 2.],
            [PI / 2., 2., 0.4, 3.33333],
            [11., 3., 0.6, -1.],
        ]);
        assert_eq!(mat / value, div_mat)
    }
    #[test]
    fn test2_matrix4div_f32() {
        let mat = Matrix4::new([
            [17.4, 9.3, 1., 0.],
            [3., 0., -12., 6.],
            [3. * PI / 2., 6., 1.2, 10.],
            [33., 9., 1.8, -3.],
        ]);
        let value = 1.;
        assert_eq!(mat / value, mat);
    }
    #[test]
    fn test3_matrix4div_f32() {
        let mat = Matrix4::new([
            [17.4, 9.3, 1., 0.],
            [3., 0., -12., 6.],
            [3. * PI / 2., 6., 1.2, 10.],
            [33., 9., 1.8, -3.],
        ]);
        let value = -1.;
        assert_eq!(mat / value, -mat);
    }
    // Impl DivAssign<f32>
    #[test]
    fn test1_matrix4div_assign_f32() {
        let mut mat = Matrix4::new([
            [17.4, 9.3, 1., 0.],
            [3., 0., -12., 6.],
            [3. * PI / 2., 6., 1.2, 10.],
            [33., 9., 1.8, -3.],
        ]);
        let value = 3.;
        let div_assign_mat = mat / value;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
    #[test]
    fn test2_matrix4div_assign_f32() {
        let mut mat = Matrix4::new([
            [17.4, 9.3, 1., 0.],
            [3., 0., -12., 6.],
            [3. * PI / 2., 6., 1.2, 10.],
            [33., 9., 1.8, -3.],
        ]);
        let value = 1.;
        let div_assign_mat = mat;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
    #[test]
    fn test3_matrix4div_assign_f32() {
        let mut mat = Matrix4::new([
            [17.4, 9.3, 1., 0.],
            [3., 0., -12., 6.],
            [3. * PI / 2., 6., 1.2, 10.],
            [33., 9., 1.8, -3.],
        ]);
        let value = -1.;
        let div_assign_mat = -mat;
        mat /= value;
        assert_eq!(mat, div_assign_mat);
    }
}
