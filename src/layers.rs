use ndarray::{Array, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, concatenate};

// Structs

struct Linear {
    weights: Array2<f32>,
    bias: Option<Array1<f32>>,
}

// Implementations

impl Linear {
    pub fn new(weights: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        Self { weights, bias }
    }

    pub fn forward(&self, x: ArrayView2<f32>) {
        let mut output = x.dot(&self.weights.t());

        if let Some(b) = &self.bias {
            output += b;
        }
    }
}
