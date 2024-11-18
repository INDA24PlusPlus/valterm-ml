use ndarray::Array1;

pub fn mse(y: &Array1<f64>, y_hat: &Array1<f64>) -> f64 {
    (y - y_hat).map(|x| x.powi(2)).sum() / y.len() as f64
}
