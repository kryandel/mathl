pub mod mat2;
pub mod mat3;
pub mod mat4;
use mat2::Matrix2;
use mat3::Matrix3;
use mat4::Matrix4;

use crate::constants::EPSILON;

pub trait Matrix
where
    Self: Sized,
    Self: Copy,
    Self: Clone,
{
    type Row;
    type Column;

    fn scalar(value: f32) -> Self;
    fn idenity() -> Self {
        Self::scalar(1.)
    }
    fn zero() -> Self {
        Self::scalar(0.)
    }

    fn dim(self) -> usize;
    fn det(self) -> f32;

    fn get_row(self, i: usize) -> Self::Row;
    fn get_col(self, j: usize) -> Self::Column;
    fn transpose(self) -> Self;

    fn try_invert(self) -> Option<Self>;
    fn invert(self) -> Self {
        self.try_invert()
            .expect("It is impossible to invert a singular matrix")
    }

    fn is_singular(self) -> bool {
        let det = self.det().abs();
        if det < EPSILON {
            return true;
        } else {
            return false;
        }
    }
}

#[allow(non_snake_case)]
pub(crate) mod transform_matrix {
    use super::*;

    pub fn scaling_matrix_in_2d(a: f32, b: f32) -> Matrix2 {
        Matrix2::new([[a, 0.], [0., b]])
    }

    pub fn scaling_matrix_in_3d(a: f32, b: f32, c: f32) -> Matrix3 {
        Matrix3::new([[a, 0., 0.], [0., b, 0.], [0., 0., c]])
    }

    pub fn scaling_matrix_in_homogeneous_2d(a: f32, b: f32) -> Matrix3 {
        Matrix3::new([[a, 0., 0.], [0., b, 0.], [0., 0., 1.]])
    }

    pub fn scaling_matrix_in_homogeneous_3d(a: f32, b: f32, c: f32) -> Matrix4 {
        Matrix4::new([
            [a, 0., 0., 0.],
            [0., b, 0., 0.],
            [0., 0., c, 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn rotation_matrix_in_2d(phi: f32) -> Matrix2 {
        let c = phi.cos();
        let s = phi.sin();
        Matrix2::new([[c, -s], [s, c]])
    }

    pub fn rotation_matrix_in_3d_Ox(phi: f32) -> Matrix3 {
        let c = phi.cos();
        let s = phi.sin();
        Matrix3::new([[1., 0., 0.], [0., c, -s], [0., s, c]])
    }

    pub fn rotation_matrix_in_3d_Oy(psi: f32) -> Matrix3 {
        let c = psi.cos();
        let s = psi.sin();
        Matrix3::new([[c, 0., s], [0., 1., 0.], [-s, 0., c]])
    }

    pub fn rotation_matrix_in_3d_Oz(xi: f32) -> Matrix3 {
        let c = xi.cos();
        let s = xi.sin();
        Matrix3::new([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    }

    pub fn rotation_matrix_in_homogeneous_2d(phi: f32) -> Matrix3 {
        let c = phi.cos();
        let s = phi.sin();
        Matrix3::new([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    }

    pub fn rotation_matrix_in_homogeneous_3d_Ox(phi: f32) -> Matrix4 {
        let c = phi.cos();
        let s = phi.sin();
        Matrix4::new([
            [1., 0., 0., 0.],
            [0., c, -s, 0.],
            [0., s, c, 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn rotation_matrix_in_homogeneous_3d_Oy(psi: f32) -> Matrix4 {
        let c = psi.cos();
        let s = psi.sin();
        Matrix4::new([
            [c, 0., s, 0.],
            [0., 1., 0., 0.],
            [-s, 0., c, 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn rotation_matrix_in_homogeneous_3d_Oz(xi: f32) -> Matrix4 {
        let c = xi.cos();
        let s = xi.sin();
        Matrix4::new([
            [c, -s, 0., 0.],
            [s, c, 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    }

    pub fn translate_matrix_in_homogeneous_2d(a: f32, b: f32) -> Matrix3 {
        Matrix3::new([[1., 0., a], [0., 1., b], [0., 0., 1.]])
    }

    pub fn translate_matrix_in_homogeneous_3d(a: f32, b: f32, c: f32) -> Matrix4 {
        Matrix4::new([
            [1., 0., 0., a],
            [0., 1., 0., b],
            [0., 0., 1., c],
            [0., 0., 0., 1.],
        ])
    }
}
