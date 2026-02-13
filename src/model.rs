use ndarray::{Array, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, concatenate};
use ndarray_conv::{ConvFFTExt, ConvMode, PaddingMode, get_fft_processor};

use crate::config::ModelDimensions;
use crate::ops::sinusoids;

// Structs

pub struct AudioEncoder {
    pub config: ModelDimensions,
}

pub struct TextDecoder {
    pub config: ModelDimensions,
}

pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
}

// Implementations

impl AudioEncoder {
    pub fn new(config: ModelDimensions) -> Self {
        let mut processor = get_fft_processor::<f32, f32>();
        let positional_embedding: Array2<f32> =
            sinusoids(config.n_audio_ctx, config.n_audio_state, 10000.);
        let blocks = [];
        Self { config }
    }

    pub fn forward(&self, mel: ArrayView3<f32>) -> Array3<f32> {
        let j = 0;
        Array3::zeros((1, 1, 1))
    }
}

impl TextDecoder {
    pub fn new(config: ModelDimensions) -> Self {
        Self { config }
    }

    pub fn forward(&self, x: ArrayView2<i64>, xa: Array3<f32>) {}
}

impl Whisper {
    pub fn new(dims: ModelDimensions) -> Self {
        let encoder = AudioEncoder::new(dims);
        let decoder = TextDecoder::new(dims);

        Self { encoder, decoder }
    }

    pub fn embed_audio(&self, mel: ArrayView3<f32>) {
        self.encoder.forward(mel);
    }

    pub fn logits(&self, tokens: ArrayView2<i64>, audio_features: Array3<f32>) {
        self.decoder.forward(tokens, audio_features);
    }

    pub fn forward(&self, mel: ArrayView3<f32>, tokens: ArrayView2<i64>) {
        self.decoder.forward(tokens, self.encoder.forward(mel));
    }
}
