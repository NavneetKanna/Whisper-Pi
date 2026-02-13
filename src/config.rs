#[derive(Clone, Copy)]
pub struct ModelDimensions {
    pub n_mels: usize,
    pub n_audio_ctx: usize,
    pub n_audio_state: usize, // embedding_dim
    pub n_audio_head: usize,
    pub n_audio_layer: usize,
    pub n_vocab: usize,
    pub n_text_ctx: usize,
    pub n_text_state: usize, // embedding_dim
    pub n_text_head: usize,
    pub n_text_layer: usize,
}
