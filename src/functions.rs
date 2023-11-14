pub mod constants {
    pub const PRECISION: usize = 3;
    pub const EPSILON: f32 = 1. / 10_i32.pow(PRECISION as _) as f32;
    pub const PI: f32 = std::f32::consts::PI;
}

pub fn unpack<const SIZE: usize>(arr: [[f32; SIZE]; SIZE]) -> Vec<f32> {
    let mut res = Vec::<f32>::default();
    for i in 0..SIZE {
        for j in 0..SIZE {
            res.push(arr[i][j]);
        }
    }
    res
}
