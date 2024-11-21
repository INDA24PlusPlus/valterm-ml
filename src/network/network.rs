use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use crate::functions::activation::Activation;

/// A neural network with generic activation function
#[derive(Debug)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
}

#[derive(Debug)]
pub enum LayerType {
    Dense {
        weights: Array2<f32>,
        biases: Array1<f32>,
    },
    Activation {
        activation: Activation,
    },
}

#[derive(Debug)]
pub struct Layer {
    pub layer_type: LayerType,

    // Cache for backpropagation
    pub inputs: Array1<f32>,
    pub outputs: Array1<f32>,
}

impl Layer {
    pub fn forward(&mut self, inputs: &Array1<f32>) -> Array1<f32> {
        self.inputs = inputs.clone();
        self.outputs = match &self.layer_type {
            LayerType::Dense { weights, biases } => weights.dot(inputs) + biases,
            LayerType::Activation { activation } => activation.activate(inputs),
        };
        self.outputs.clone()
    }

    pub fn backward(&self, delta: &Array1<f32>) -> Array1<f32> {
        match &self.layer_type {
            LayerType::Dense { weights, .. } => weights.t().dot(delta),
            LayerType::Activation { activation } => activation.derivative(delta),
        }
    }

    pub fn dense(inputs: usize, outputs: usize) -> Layer {
        let weights = Array2::random((outputs, inputs), Uniform::new(-1.0, 1.0));
        let biases = Array1::random(outputs, Uniform::new(-1.0, 1.0));

        Layer {
            layer_type: LayerType::Dense { weights, biases },
            inputs: Array1::zeros(inputs),
            outputs: Array1::zeros(outputs),
        }
    }

    pub fn activation(activation: Activation) -> Layer {
        Layer {
            layer_type: LayerType::Activation { activation },
            inputs: Array1::zeros(0),
            outputs: Array1::zeros(0),
        }
    }
}

/// A layer in a neural network with generic activation function

impl NeuralNetwork {
    pub fn new(layers: Vec<Layer>) -> NeuralNetwork {
        NeuralNetwork { layers }
    }

    pub fn predict(&mut self, inputs: Array1<f32>) -> Array1<f32> {
        self.layers
            .iter_mut()
            .fold(inputs, |acc, layer| layer.forward(&acc))
    }
}
