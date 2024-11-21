use std::vec;

use dataset::mnist::{format_labels, Mnist};
use functions::activation::{Activation, ActivationType};
use network::network::{Layer, NeuralNetwork};
use training::backprop::Trainer;
use util::get_accuracy;

pub mod dataset;
pub mod functions;
pub mod network;
pub mod training;
pub mod util;

fn main() {
    let mnist = Mnist::get();

    let mut nn = NeuralNetwork::new(vec![
        Layer::dense(28 * 28, 10),
        Layer::activation(Activation::new(ActivationType::ReLU)),
        Layer::dense(10, 10),
        Layer::activation(Activation::new(ActivationType::ReLU)),
        Layer::dense(10, 10),
        Layer::activation(Activation::new(ActivationType::Softmax)),
    ]);

    println!(
        "Accuracy: {}",
        get_accuracy(
            mnist.test.clone(),
            format_labels(&mnist.test_labels),
            &mut nn
        )
    );

    let mut trainer = Trainer::new(&mut nn, 0.01);

    let targets: Vec<_> = mnist
        .train_labels
        .iter()
        .map(|&x| dataset::mnist::format_label(x))
        .collect();

    trainer.train(mnist.train.clone(), targets, 3);

    println!(
        "Accuracy: {}",
        get_accuracy(
            mnist.test.clone(),
            format_labels(&mnist.test_labels),
            &mut nn
        )
    );
}
