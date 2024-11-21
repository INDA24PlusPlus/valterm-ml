use crate::functions::activation::{Activation, Softmax};
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// A neural network with generic activation function
#[derive(Debug)]
pub struct NeuralNetwork<A: Activation> {
    pub layers: Vec<Layer<A>>,
}

/// A layer in a neural network with generic activation function
#[derive(Debug)]
pub struct Layer<A: Activation> {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub output: bool,
    pub activation: A,
}

impl<A: Activation> Layer<A> {
    pub fn new(weights: Array2<f32>, biases: Array1<f32>, activation: A) -> Layer<A> {
        Layer {
            weights,
            biases,
            activation,
            output: false,
        }
    }

    pub fn random(input_size: usize, output_size: usize, activation: A, output: bool) -> Layer<A> {
        let weights = Array::random((output_size, input_size), Uniform::new(-1.0, 1.0));
        let biases = Array::random(output_size, Uniform::new(-1.0, 1.0));
        Layer {
            weights,
            biases,
            activation,
            output,
        }
    }

    /// Compute the output of the layer given the input, will apply the activation function
    pub fn feedforward(&self, inputs: Array1<f32>) -> Array1<f32> {
        let z = self.weights.dot(&inputs) + &self.biases;
        if self.output {
            return Softmax.activate(&z);
        }
        self.activation.activate(&z)
    }
}

impl<A: Activation> NeuralNetwork<A> {
    pub fn new(layers: Vec<Layer<A>>) -> NeuralNetwork<A> {
        NeuralNetwork { layers }
    }

    /// Create a new neural network with random weights and biases
    pub fn random(
        input_size: usize,
        output_size: usize,
        layer_sizes: &[usize],
        activation: A,
    ) -> NeuralNetwork<A> {
        let mut layers = Vec::new();
        let mut input_size = input_size;
        for &layer_size in layer_sizes {
            let layer = Layer::random(input_size, layer_size, activation, false);
            layers.push(layer);
            input_size = layer_size;
        }
        let output_layer = Layer::random(input_size, output_size, activation, true);
        layers.push(output_layer);
        NeuralNetwork::new(layers)
    }

    pub fn predict(&self, inputs: Array1<f32>) -> Array1<f32> {
        self.layers
            .iter()
            .fold(inputs, |acc, layer| layer.feedforward(acc))
    }
}
