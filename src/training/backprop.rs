use ndarray::{Array1, Array2, Axis};

use crate::functions::error::cross_entropy_error;
use crate::network::network::{LayerType, NeuralNetwork};
use crate::util::{create_batch, get_accuracy_bruh};

const BATCH_SIZE: usize = 32;

pub struct Trainer<'a> {
    network: &'a mut NeuralNetwork,
    learning_rate: f32,
}

impl<'a> Trainer<'a> {
    pub fn new(network: &'a mut NeuralNetwork, learning_rate: f32) -> Trainer<'a> {
        Trainer {
            network,
            learning_rate,
        }
    }

    // Single epoch of training
    pub fn backprop(&mut self, inputs: &Array2<f32>, targets: &Array2<f32>) {
        let out = self.network.predict(inputs.clone());
        //println!("Predicted: {:?}", out);

        // Initialize dE/dY
        //let mut delta = mse_derivative(&out, targets);
        let mut delta = &out - targets;
        let _error = cross_entropy_error(&out, targets);
        /* println!("Predicted: {:?}", out);
        println!("Target: {:?}", targets); */
        //println!("Error: {}", _error);

        // Backpropagate the error
        for layer in self.network.layers.iter_mut().rev() {
            let delta_new = layer.backward(&delta);

            // Update the weights and biases
            if let LayerType::Dense { weights, biases } = &mut layer.layer_type {
                /* println!("=== Layer ===");
                println!("Weights shape: {:?}", weights.shape());
                println!("Delta shape: {:?}", delta.shape());
                println!("Inputs shape: {:?}", layer.inputs.shape());
                println!("Inputs shaped: {:?}", inputs_shaped.shape());
                println!("Delta shaped: {:?}", delta_shaped.shape());
                println!("dw shape: {:?}", dw.shape());
                println!("=== Layer ==="); */

                // dw has to be same shape as weights (m x n) => Inputs (n x 1) * Delta (1 x m)
                /* let inputs_shaped = layer.inputs.to_shape((layer.inputs.len(), 1)).unwrap();
                let delta_shaped = delta.to_shape((1, delta.len())).unwrap(); */

                let dw = layer.inputs.t().dot(&delta).to_owned();

                let db = delta.clone().sum_axis(Axis(0));

                *weights = weights.clone() - (self.learning_rate * dw);
                *biases = biases.clone() - (self.learning_rate * db);
            }

            delta = delta_new;
        }
    }

    pub fn train(&mut self, inputs: Vec<Array1<f32>>, targets: Vec<Array1<f32>>, epochs: usize) {
        // Turn 1D arrays into 2D arrays with batches
        let inputs_batched = create_batch(BATCH_SIZE, inputs);
        let targets_batched = create_batch(BATCH_SIZE, targets);

        for i in 0..epochs {
            println!("Beginning epoch {}", i);
            for (inputs, targets) in inputs_batched.iter().zip(targets_batched.iter()) {
                self.backprop(inputs, targets);
            }

            let acc = get_accuracy_bruh(
                inputs_batched.clone(),
                targets_batched.clone(),
                self.network,
            );
            println!("Accuracy: {}", acc);
        }
    }
}
