use functions::activation::ReLU;
use network::network::NeuralNetwork;
use training::backprop::Trainer;

pub mod dataset;
pub mod functions;
pub mod network;
pub mod training;

fn main() {
    //let mnist = dataset::mnist::Mnist::get();

    let network = NeuralNetwork::random(2, 2, &[4, 4], ReLU);
    println!("{:?}", network.predict(vec![1.0, 2.0].into()));
    let mut trainer = Trainer::new(network, 0.1);
    trainer.backprop(vec![1.0, 2.0].into(), vec![0.0, 1.0].into());
}
