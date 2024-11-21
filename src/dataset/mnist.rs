use ndarray::Array1;

pub const TRAIN_SIZE: usize = 60_000; // 60_000
pub const TEST_SIZE: usize = 10_000;

pub struct Mnist {
    pub train_labels: Vec<u8>,
    pub train: Vec<Array1<f32>>,
    pub test: Vec<Array1<f32>>,
    pub test_labels: Vec<u8>,
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
            train_labels: mnist.trn_lbl.to_vec(),
            test,
            test_labels: mnist.tst_lbl.to_vec(),
        }
    }
}

fn format_img(img: &[u8], samples: usize) -> Vec<Array1<f32>> {
    // Reshape the images to a vector of 1D arrays
    img.chunks(28 * 28)
        .take(samples)
        .map(|chunk| {
            let mut arr = Array1::zeros(28 * 28);
            for (i, &x) in chunk.iter().enumerate() {
                arr[i] = x as f32 / 256.0;
            }

            arr
        })
        .collect()
}

pub fn format_label(label: u8) -> Array1<f32> {
    let mut arr = Array1::zeros(10);
    arr[label as usize] = 1.;

    arr
}

pub fn format_labels(labels: &Vec<u8>) -> Vec<Array1<f32>> {
    labels.iter().map(|&x| format_label(x)).collect()
}
