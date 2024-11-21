use ndarray::{Array1, Array2};

pub trait Activation: Clone + Copy + Default {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32>;
    fn derivative(&self, x: &Array1<f32>) -> Array1<f32>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax;

impl Activation for Sigmoid {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        let s = self.activate(x);
        s.mapv(|v| v * (1.0 - v))
    }
}

impl Activation for ReLU {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

impl Activation for Softmax {
    fn activate(&self, x: &Array1<f32>) -> Array1<f32> {
        let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps = x.map(|v| (v - max).exp());
        let sum: f32 = exps.iter().sum();
        exps / sum
    }

    // ChatGPT helped me with this one :)
    fn derivative(&self, x: &Array1<f32>) -> Array1<f32> {
        let s = self.activate(x);
        let mut jacobian = Array2::zeros((s.len(), s.len()));
        for i in 0..s.len() {
            for j in 0..s.len() {
                jacobian[[i, j]] = if i == j {
                    s[i] * (1.0 - s[i])
                } else {
                    -s[i] * s[j]
                };
            }
        }
        jacobian.diag().to_owned()
    }
}
