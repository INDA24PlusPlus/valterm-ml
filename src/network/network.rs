use crate::functions::activation::Activation;
use ndarray::prelude::*;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// A neural network with generic activation function
#[derive(Debug)]
pub struct NeuralNetwork<A: Activation> {
    layers: Vec<Layer<A>>,
}

/// A layer in a neural network with generic activation function
// TODO: Handle activation differently in last layer (?)
#[derive(Debug)]
pub struct Layer<A: Activation> {
    weights: Array2<f64>,
    biases: Array1<f64>,
    activation: A,
}

impl<A: Activation> Layer<A> {
    pub fn new(weights: Array2<f64>, biases: Array1<f64>, activation: A) -> Layer<A> {
        Layer {
            weights,
            biases,
            activation,
        }
    }

    pub fn random(input_size: usize, output_size: usize, activation: A) -> Layer<A> {
        let weights = Array::random((output_size, input_size), Uniform::new(-1.0, 1.0));
        let biases = Array::random(output_size, Uniform::new(-1.0, 1.0));
        Layer {
            weights,
            biases,
            activation,
        }
    }

    /// Compute the output of the layer given the input, will apply the activation function
    pub fn feedforward(&self, inputs: Array1<f64>) -> Array1<f64> {
        let z = self.weights.dot(&inputs) + &self.biases;
        z.map(|x| self.activation.activate(*x))
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
            let layer = Layer::random(input_size, layer_size, activation);
            layers.push(layer);
            input_size = layer_size;
        }
        let output_layer = Layer::random(input_size, output_size, activation);
        layers.push(output_layer);
        NeuralNetwork::new(layers)
    }

    pub fn predict(&self, inputs: Array1<f64>) -> Array1<f64> {
        self.layers
            .iter()
            .fold(inputs, |acc, layer| layer.feedforward(acc))
    }
}
