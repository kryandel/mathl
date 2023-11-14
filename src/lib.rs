pub(crate) mod functions;
pub(crate) mod matrices;
pub(crate) mod vectors;

// pub (crate) mod quaternions;

use std::ops::Mul;

pub use mat2::Matrix2;
pub use mat3::Matrix3;
pub use mat4::Matrix4;
pub use matrices::Matrix;
use matrices::*;
use transform_matrix::*;

pub use vec2::Vector2;
pub use vec3::Vector3;
pub use vec4::Vector4;
pub use vectors::Vector;
use vectors::*;

pub use functions::constants;

impl Mul<Vector2> for Matrix2 {
    type Output = Vector2;

    fn mul(self, rhs: Vector2) -> Self::Output {
        Vector2::new(
            self[(0, 0)] * rhs.x + self[(0, 1)] * rhs.y,
            self[(1, 0)] * rhs.x + self[(1, 1)] * rhs.y,
        )
    }
}
impl Mul<Vector3> for Matrix3 {
    type Output = Vector3;

    fn mul(self, rhs: Vector3) -> Self::Output {
        Vector3::new(
            self[(0, 0)] * rhs.x + self[(0, 1)] * rhs.y + self[(0, 2)] * rhs.z,
            self[(1, 0)] * rhs.x + self[(1, 1)] * rhs.y + self[(1, 2)] * rhs.z,
            self[(2, 0)] * rhs.x + self[(2, 1)] * rhs.y + self[(2, 2)] * rhs.z,
        )
    }
}
impl Mul<Vector4> for Matrix4 {
    type Output = Vector4;

    fn mul(self, rhs: Vector4) -> Self::Output {
        Vector4::new(
            self[(0, 0)] * rhs.x
                + self[(0, 1)] * rhs.y
                + self[(0, 2)] * rhs.z
                + self[(0, 3)] * rhs.w,
            self[(1, 0)] * rhs.x
                + self[(1, 1)] * rhs.y
                + self[(1, 2)] * rhs.z
                + self[(1, 3)] * rhs.w,
            self[(2, 0)] * rhs.x
                + self[(2, 1)] * rhs.y
                + self[(2, 2)] * rhs.z
                + self[(2, 3)] * rhs.w,
            self[(3, 0)] * rhs.x
                + self[(3, 1)] * rhs.y
                + self[(3, 2)] * rhs.z
                + self[(3, 3)] * rhs.w,
        )
    }
}

pub mod transform {

    use super::*;

    pub fn scale2(coefficients: (f32, f32)) -> Matrix2 {
        let (a, b) = coefficients;
        scaling_matrix_in_2d(a, b)
    }

    pub fn scale2x(coefficient: f32) -> Matrix2 {
        scale2((coefficient, 1.))
    }

    pub fn scale2y(coefficient: f32) -> Matrix2 {
        scale2((1., coefficient))
    }

    pub fn scale3(coefficients: (f32, f32, f32)) -> Matrix3 {
        let (a, b, c) = coefficients;
        scaling_matrix_in_3d(a, b, c)
    }

    pub fn scale3x(coefficient: f32) -> Matrix3 {
        scale3((coefficient, 1., 1.))
    }

    pub fn scale3y(coefficient: f32) -> Matrix3 {
        scale3((1., coefficient, 1.))
    }

    pub fn scale3z(coefficient: f32) -> Matrix3 {
        scale3((1., 1., coefficient))
    }

    pub fn rotate2(phi: f32) -> Matrix2 {
        rotation_matrix_in_2d(phi)
    }

    pub fn rotate3x(phi: f32) -> Matrix3 {
        rotation_matrix_in_3d_Ox(phi)
    }

    pub fn rotate3y(psi: f32) -> Matrix3 {
        rotation_matrix_in_3d_Oy(psi)
    }

    pub fn rotate3z(xi: f32) -> Matrix3 {
        rotation_matrix_in_3d_Oz(xi)
    }

    pub fn rotate3(phi: f32, psi: f32, xi: f32) -> Matrix3 {
        rotate3z(xi) * rotate3x(phi) * rotate3y(psi)
    }

    pub mod homogeneous {

        use super::*;
        use constants::PI;

        pub fn scale2(coefficients: Vector2) -> Matrix3 {
            let Vector2 { x: a, y: b } = coefficients;
            scaling_matrix_in_homogeneous_2d(a, b)
        }

        pub fn scale2x(coefficient: f32) -> Matrix3 {
            scale2(Vector2::new(coefficient, 1.))
        }

        pub fn scale2y(coefficient: f32) -> Matrix3 {
            scale2(Vector2::new(1., coefficient))
        }

        pub fn scale3(coefficients: Vector3) -> Matrix4 {
            let Vector3 { x: a, y: b, z: c } = coefficients;
            scaling_matrix_in_homogeneous_3d(a, b, c)
        }

        pub fn scale3x(coefficient: f32) -> Matrix4 {
            scale3(Vector3::new(coefficient, 1., 1.))
        }

        pub fn scale3y(coefficient: f32) -> Matrix4 {
            scale3(Vector3::new(1., coefficient, 1.))
        }

        pub fn scale3z(coefficient: f32) -> Matrix4 {
            scale3(Vector3::new(1., 1., coefficient))
        }

        pub fn rotate2(phi: f32) -> Matrix3 {
            rotation_matrix_in_homogeneous_2d(phi)
        }

        pub fn rotate3x(phi: f32) -> Matrix4 {
            rotation_matrix_in_homogeneous_3d_Ox(phi)
        }

        pub fn rotate3y(psi: f32) -> Matrix4 {
            rotation_matrix_in_homogeneous_3d_Oy(psi)
        }

        pub fn rotate3z(xi: f32) -> Matrix4 {
            rotation_matrix_in_homogeneous_3d_Oz(xi)
        }

        pub fn rotate3(phi: f32, psi: f32, xi: f32) -> Matrix4 {
            rotate3z(xi) * rotate3x(phi) * rotate3y(psi)
        }

        pub fn translate2(coefficients: Vector2) -> Matrix3 {
            let Vector2 { x: a, y: b } = coefficients;
            translate_matrix_in_homogeneous_2d(a, b)
        }

        pub fn translate2x(coefficient: f32) -> Matrix3 {
            translate2(Vector2::new(coefficient, 0.))
        }

        pub fn translate2y(coefficient: f32) -> Matrix3 {
            translate2(Vector2::new(0., coefficient))
        }

        pub fn translate3(coefficients: Vector3) -> Matrix4 {
            let Vector3 { x: a, y: b, z: c } = coefficients;
            translate_matrix_in_homogeneous_3d(a, b, c)
        }

        pub fn translate3x(coefficient: f32) -> Matrix4 {
            translate3(Vector3::new(coefficient, 0., 0.))
        }

        pub fn translate3y(coefficient: f32) -> Matrix4 {
            translate3(Vector3::new(0., coefficient, 0.))
        }

        pub fn translate3z(coefficient: f32) -> Matrix4 {
            translate3(Vector3::new(0., 0., coefficient))
        }

        pub fn perspective3(z_far: f32, z_near: f32, aspect_ratio: f32, fov: f32) -> Matrix4 {
            let f = (PI * 0.5 - 0.5 * fov).tan();
            let range_inverse = 1. / (z_near - z_far);
            Matrix4::new([
                [f / (aspect_ratio), 0., 0., 0.],
                [0., f, 0., 0.],
                [
                    0.,
                    0.,
                    (z_near + z_far) * range_inverse,
                    z_near * z_far * range_inverse * 2.,
                ],
                [0., 0., -1., 0.],
            ])
        }

        pub fn ortho3(
            left: f32,
            right: f32,
            bottom: f32,
            top: f32,
            near_val: f32,
            far_val: f32,
        ) -> Matrix4 {
            let rl_range = right - left;
            let tb_range = top - bottom;
            let val_range = far_val - near_val;
            let t_x = -(right + left) / rl_range;
            let t_y = -(top + bottom) / tb_range;
            let t_z = -(far_val + near_val) / val_range;
            Matrix4::new([
                [2. / rl_range, 0., 0., t_x],
                [0., 2. / tb_range, 0., t_y],
                [0., 0., -2. / val_range, t_z],
                [0., 0., 0., 1.],
            ])
        }

        #[allow(non_snake_case)]
        pub fn lookat3(camera: Vector3, target: Vector3, world_up: Vector3) -> Matrix4 {
            let direction = (camera - target).normalize();
            let camera_right = world_up.cross(direction).normalize();
            let camera_up = direction.cross(camera_right);
            let Vector3 {
                x: D_x,
                y: D_y,
                z: D_z,
            } = direction;
            let Vector3 {
                x: R_x,
                y: R_y,
                z: R_z,
            } = camera_right;
            let Vector3 {
                x: U_x,
                y: U_y,
                z: U_z,
            } = camera_up;
            Matrix4::new([
                [R_x, R_y, R_z, 0.],
                [U_x, U_y, U_z, 0.],
                [D_x, D_y, D_z, 0.],
                [0.0, 0.0, 0.0, 1.],
            ]) * translate3(-camera)
        }
    }
}
