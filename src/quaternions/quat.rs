// use crate::vectors::vec3::Vector3;

// #[derive(Default, Debug, Clone, Copy)]
// struct Quaternion {
//     a: f32,
//     i: f32,
//     j: f32,
//     k: f32,
// }
// impl Quaternion {
//     fn new(a: f32, i: f32, j: f32, k: f32) -> Self {
//         Quaternion { a, i, j, k }
//     }
// }
// impl From<(f32, Vector3)> for Quaternion {
//     fn from(value: (f32, Vector3)) -> Self {
//         let (a, i, j, k) = (value.0, value.1.x, value.1.y, value.1.z);
//         Quaternion::new(a, i, j, k)
//     }
// }
// impl Into<(f32, Vector3)> for Quaternion {
//     fn into(self) -> (f32, Vector3) {
//         (self.a, Vector3::new(self.i, self.j, self.k))
//     }
// }
