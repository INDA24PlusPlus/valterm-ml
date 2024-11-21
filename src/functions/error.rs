use ndarray::Array1;

pub fn mse(y: &Array1<f32>, y_hat: &Array1<f32>) -> f32 {
    (y - y_hat).map(|x| x.powi(2)).sum() / y.len() as f32
}

pub fn mse_derivative(y: &Array1<f32>, y_hat: &Array1<f32>) -> Array1<f32> {
    2.0 * (y - y_hat) / y.len() as f32
}
