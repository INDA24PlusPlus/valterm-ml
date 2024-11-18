use ndarray::Array1;

pub trait Activation: Clone + Copy + Default {
    fn activate(&self, x: f64) -> f64;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;

impl Activation for Sigmoid {
    fn activate(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
}

impl Activation for ReLU {
    fn activate(&self, x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps = x.map(|v| (v - max).exp());
    let sum: f64 = exps.iter().sum();
    exps / sum
}
