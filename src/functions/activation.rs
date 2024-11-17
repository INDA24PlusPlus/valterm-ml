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
