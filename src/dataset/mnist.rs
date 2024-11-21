use ndarray::{Array1, Array2};

pub const TRAIN_SIZE: usize = 60_000;
pub const TEST_SIZE: usize = 10_000;

pub struct Mnist {
    pub train: Vec<Array1<f32>>,
    pub labels: Vec<u8>,
    pub test: Vec<Array1<f32>>,
}

impl Mnist {
    pub fn get() -> Mnist {
        let mnist = mnist::MnistBuilder::new()
            .label_format_digit()
            .training_set_length(TRAIN_SIZE as u32)
            .test_set_length(TEST_SIZE as u32)
            .finalize();

        let train = format_img(&mnist.trn_img, TRAIN_SIZE);
        let test = format_img(&mnist.tst_img, TEST_SIZE);

        Mnist {
            train,
            labels: mnist.trn_lbl.to_vec(),
            test,
        }
    }
}

fn format_img(img: &[u8], samples: usize) -> Vec<Array1<f32>> {
    let img = Array2::from_shape_vec((28 * 28, samples), img.to_vec())
        .expect("Failed to reshape image data")
        .map(|x| *x as f32 / 256.);

    img.columns().into_iter().map(|x| x.to_owned()).collect()
}
