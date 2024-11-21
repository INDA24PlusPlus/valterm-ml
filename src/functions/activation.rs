use ndarray::{Array1, Array2};

#[derive(Debug)]
pub enum ActivationType {
    Sigmoid,
    ReLU,
    Softmax,
}

#[derive(Debug)]
pub struct Activation {
    pub activation: ActivationType,
}

impl Activation {
    pub fn new(activation: ActivationType) -> Activation {
        Activation { activation }
    }

    pub fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            ActivationType::Sigmoid => Sigmoid.activate(x),
            ActivationType::ReLU => ReLU.activate(x),
            ActivationType::Softmax => Softmax.activate(x),
        }
    }

    pub fn derivative(&self, x: &Array2<f32>, inputs: &Array2<f32>) -> Array2<f32> {
        match self.activation {
            ActivationType::Sigmoid => x * Sigmoid.derivative(inputs),
            ActivationType::ReLU => x * ReLU.derivative(inputs),
            ActivationType::Softmax => Softmax.derivative(x),
        }
    }
}

pub trait ActivationTrait: Clone + Copy + Default {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32>;
    fn derivative(&self, x: &Array2<f32>) -> Array2<f32>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Sigmoid;
#[derive(Debug, Clone, Copy, Default)]
pub struct ReLU;
#[derive(Debug, Clone, Copy, Default)]
pub struct Softmax;

impl ActivationTrait for Sigmoid {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| 1.0 / (1.0 + (-v).exp()))
    }

    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        let s = self.activate(x);
        s.mapv(|v| v * (1.0 - v))
    }
}

impl ActivationTrait for ReLU {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| if v > 0.0 { v } else { 0.0 })
    }

    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        x.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 })
    }
}

fn softmax_1d(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps = x.map(|v| (v - max).exp());
    let sum: f32 = exps.iter().sum();
    exps / sum
}

impl ActivationTrait for Softmax {
    fn activate(&self, x: &Array2<f32>) -> Array2<f32> {
        let mut softmax = Array2::zeros(x.raw_dim());
        for (i, row) in x.outer_iter().enumerate() {
            softmax.row_mut(i).assign(&softmax_1d(&row.to_owned()));
        }
        softmax
    }

    // ChatGPT helped me with this one :) fuck chat gpt liam i love you
    fn derivative(&self, x: &Array2<f32>) -> Array2<f32> {
        x.clone()
    }
}
