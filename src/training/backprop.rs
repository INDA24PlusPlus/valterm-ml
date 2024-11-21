use ndarray::Array1;

use crate::functions::activation::{Activation, Softmax};
use crate::functions::error::{mse, mse_derivative};
use crate::network::network::NeuralNetwork;

pub struct Trainer<A: Activation> {
    network: NeuralNetwork<A>,
    learning_rate: f32,
}

impl<A: Activation> Trainer<A> {
    pub fn new(network: NeuralNetwork<A>, learning_rate: f32) -> Trainer<A> {
        Trainer {
            network,
            learning_rate,
        }
    }

    // https://towardsdatascience.com/backpropagation-made-easy-e90a4d5ede55
    // https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2020/linear-backprop.html
    // Single epoch of training
    pub fn backprop(&mut self, inputs: Array1<f32>, targets: Array1<f32>) {}
}
