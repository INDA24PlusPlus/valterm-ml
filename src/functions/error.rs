use ndarray::Array2;

pub fn cross_entropy_error(prediction: &Array2<f32>, target: &Array2<f32>) -> f32 {
    let epsilon = 1e-10; // avoid log(0)
    let size = prediction.dim().0;

    let err: f32 = (0..size)
        .map(|i| {
            let log_predicted = prediction.row(i).mapv(|x| (x.max(epsilon)).ln());
            -target.row(i).dot(&log_predicted)
        })
        .sum();

    err / size as f32
}
