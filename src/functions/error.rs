use ndarray::{Array1, Array2};

/* pub fn mse(y: &Array1<f32>, y_hat: &Array1<f32>) -> f32 {
    (y - y_hat).map(|x| x.powi(2)).sum() / y.len() as f32
}

pub fn mse_derivative(y: &Array1<f32>, y_hat: &Array1<f32>) -> Array1<f32> {
    2.0 * (y - y_hat) / y.len() as f32
}
 */

/* pub fn mse(y: &Array2<f32>, y_hat: &Array2<f32>) -> f32 {
    // Mean squared error for a batch of samples
    y.iter()
        .zip(y_hat.iter())
        .map(|(y, y_hat)| (y - y_hat).powi(2))
        .sum::<f32>()
        / y.len() as f32
}

pub fn mse_derivative(y: &Array2<f32>, y_hat: &Array2<f32>) -> Array2<f32> {
    // Derivative of mean squared error for a batch of samples
    2.0 * (y - y_hat) / y.len() as f32
} */

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
