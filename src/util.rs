use ndarray::{Array1, Array2};

use crate::network::network::NeuralNetwork;

pub fn create_batch(size: usize, data: Vec<Array1<f32>>) -> Vec<Array2<f32>> {
    let mut batches = Vec::new();

    for i in 0..data.len() / size {
        let batch = Array2::from_shape_vec(
            (size, data[0].len()),
            data[i * size..(i + 1) * size]
                .iter()
                .flat_map(|x| x.iter())
                .cloned()
                .collect(),
        )
        .unwrap();
        batches.push(batch);
    }

    batches
}

pub fn get_accuracy_bruh(
    batches: Vec<Array2<f32>>,
    labels: Vec<Array2<f32>>,
    network: &mut NeuralNetwork,
) -> f32 {
    // Calculate the accuracy of the network, given a batch of inputs and labels
    let mut correct = 0;
    let mut total = 0;

    for (batch, label) in batches.iter().zip(labels.iter()) {
        let predictions = network.predict(batch.clone());

        for (prediction, target) in predictions.outer_iter().zip(label.outer_iter()) {
            let pred = prediction
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0;
            let targ = target
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0;

            if pred == targ {
                correct += 1;
            }
            total += 1;
        }
    }

    correct as f32 / total as f32
}

pub fn get_accuracy(
    imgs: Vec<Array1<f32>>,
    labels: Vec<Array1<f32>>,
    network: &mut NeuralNetwork,
) -> f32 {
    let imgs_batch = create_batch(32, imgs);
    let labels_batch = create_batch(32, labels);

    get_accuracy_bruh(imgs_batch, labels_batch, network)
}
