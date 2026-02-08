struct ModelDimensions {
    n_mels: i32,
    n_audio_ctx: i32,
    n_audio_state: i32,
    n_audio_head: i32,
    n_audio_layer: i32,
    n_vocab: i32,
    n_text_ctx: i32,
    n_text_state: i32,
    n_text_head: i32,
    n_text_layer: i32
}

struct AudioEncoder {
    n_mels: i32,
    n_ctx: i32,
    n_state: i32,
    n_head: i32,
    n_layer: i32
}

struct TextDecoder{
    n_vocab: i32,
    n_ctx: i32,
    n_state: i32,
    n_head: i32,
    n_layer: i32
}


struct Whisper<'a> {
    dims: &'a ModelDimensions,
    encoder: &'a AudioEncoder,
    decoder: &'a TextDecoder
}


fn main() {
}
