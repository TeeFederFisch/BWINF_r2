use rand::Rng;

pub fn get_new(min: f32, max: f32) -> (f32, f32) {
    let mut rng = rand::thread_rng();
    return (rng.gen_range(min..max), rng.gen_range(min..max));
}

pub fn get_amount(min: f32, max: f32, amount: usize) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for _ in 0..amount {out.push(get_new(min, max));};
    out
}