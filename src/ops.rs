use ndarray::{Array, Array1, Array2, ArrayView2, Axis, concatenate};

pub fn sinusoids(length: usize, channels: usize, max_timescale: f32) -> Array2<f32> {
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
