use std::vec;

use dataset::mnist::Mnist;
use functions::activation::{Activation, ActivationType};
use network::network::{Layer, NeuralNetwork};
use training::backprop::Trainer;

pub mod dataset;
pub mod functions;
pub mod network;
pub mod training;

pub fn get_accuracy(nn: &mut NeuralNetwork, mnist: &Mnist) -> f32 {
    let mut correct = 0;

    for (inputs, &label) in mnist.test.iter().zip(mnist.test_labels.iter()) {
        let out = nn.predict(inputs.clone());
        let guess = out
            .iter()
            .position(|&x| x == *out.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
            .unwrap();

        if guess == label as usize {
            correct += 1;
        }
    }

    correct as f32 / mnist.test.len() as f32
}

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

    println!("Accuracy: {}", get_accuracy(&mut nn, &mnist));

    let mut trainer = Trainer::new(&mut nn, 0.01);

    let targets: Vec<_> = mnist
        .train_labels
        .iter()
        .map(|&x| dataset::mnist::format_label(x))
        .collect();

    trainer.train(mnist.train.clone(), targets, 3);

    println!("Accuracy: {}", get_accuracy(&mut nn, &mnist));
}
