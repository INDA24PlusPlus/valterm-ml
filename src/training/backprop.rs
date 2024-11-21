use ndarray::Array1;

use crate::functions::error::{mse, mse_derivative};
use crate::network::network::{LayerType, NeuralNetwork};

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
    pub fn backprop(&mut self, inputs: &Array1<f32>, targets: &Array1<f32>) {
        let out = self.network.predict(inputs.clone());

        // Initialize dE/dY
        let mut delta = mse_derivative(&out, targets);
        let _error = mse(&out, targets);
        /* println!("Predicted: {:?}", out);
        println!("Target: {:?}", targets); */
        //println!("Error: {}", error);

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
                let inputs_shaped = layer.inputs.to_shape((layer.inputs.len(), 1)).unwrap();
                let delta_shaped = delta.to_shape((1, delta.len())).unwrap();

                let dw = inputs_shaped.dot(&delta_shaped).t().to_owned();

                let db = delta.clone();

                *weights = weights.clone() - (self.learning_rate * dw);
                *biases = biases.clone() - (self.learning_rate * db);
            }

            delta = delta_new;
        }
    }

    pub fn train(&mut self, inputs: Vec<Array1<f32>>, targets: Vec<Array1<f32>>, epochs: usize) {
        for i in 0..epochs {
            println!("Beginning epoch {}", i);
            for (input, target) in inputs.iter().zip(targets.iter()) {
                self.backprop(input, target);
            }
        }
    }
}
