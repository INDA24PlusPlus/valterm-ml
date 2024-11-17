use functions::activation::ReLU;
use network::network::NeuralNetwork;

pub mod functions;
pub mod network;

fn main() {
    let network = NeuralNetwork::random(2, 2, &[4, 4], ReLU);
    println!("{:?}", network.predict(vec![1.0, 2.0].into()));
}
