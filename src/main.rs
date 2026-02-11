use ndarray::{Array, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis, concatenate};
use ndarray_conv::{ConvFFTExt, ConvMode, PaddingMode, get_fft_processor};

#[derive(Clone, Copy)]
struct ModelDimensions {
    n_mels: usize,
    n_audio_ctx: usize,
    n_audio_state: usize, // embedding_dim
    n_audio_head: usize,
    n_audio_layer: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_state: usize, // embedding_dim
    n_text_head: usize,
    n_text_layer: usize,
}

struct AudioEncoder {
    config: ModelDimensions,
}

struct TextDecoder {
    config: ModelDimensions,
}

struct Whisper {
    encoder: AudioEncoder,
    decoder: TextDecoder,
}

fn sinusoids(length: usize, channels: usize, max_timescale: f32) -> Array2<f32> {
    let half_channels = channels / 2;

    assert!(channels as f32 % 2.0 == 0.0);

    let log_timescale_increment: f32 = max_timescale.ln() / (half_channels - 1) as f32;

    let inv_timescales: Array1<f32> = Array::range(0., half_channels as f32, 1.)
        .mapv_into(|x| (x * -log_timescale_increment).exp());

    let positions: Array1<f32> = Array::range(0., length as f32, 1.);

    let scaled_time: Array2<f32> =
        positions.insert_axis(Axis(1)) * inv_timescales.insert_axis(Axis(0));

    let sin_part = scaled_time.mapv(f32::sin);
    let cos_part = scaled_time.mapv(f32::cos);

    concatenate(Axis(1), &[sin_part.view(), cos_part.view()]).unwrap()
}

impl AudioEncoder {
    fn new(config: ModelDimensions) -> Self {
        let mut processor = get_fft_processor::<f32, f32>();
        Self { config }
    }

    fn forward(&self, mel: ArrayView3<f32>) -> Array3<f32> {
        let j = 0;
        Array3::zeros((1, 1, 1))
    }
}

impl TextDecoder {
    fn new(config: ModelDimensions) -> Self {
        Self { config }
    }

    fn forward(&self, x: ArrayView2<i64>, xa: Array3<f32>) {}
}

impl Whisper {
    fn new(dims: ModelDimensions) -> Self {
        let encoder = AudioEncoder::new(dims);
        let decoder = TextDecoder::new(dims);

        Self { encoder, decoder }
    }

    fn embed_audio(&self, mel: ArrayView3<f32>) {
        self.encoder.forward(mel);
    }

    fn logits(&self, tokens: ArrayView2<i64>, audio_features: Array3<f32>) {
        self.decoder.forward(tokens, audio_features);
    }

    fn forward(&self, mel: ArrayView3<f32>, tokens: ArrayView2<i64>) {
        self.decoder.forward(tokens, self.encoder.forward(mel));
    }
}

fn main() {}
