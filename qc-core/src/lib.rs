use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::fmt::Debug;

fn rng(seed: u64) -> impl Rng {
    let rng: StdRng = SeedableRng::seed_from_u64(seed);
    rng
}

pub fn gen_f64(size: usize, seed: u64) -> f64 {
    rng(seed).gen::<f64>() % size as f64
}

// towards zero strategy.
pub fn shrink_f64(x: &f64) -> Option<f64> {
    if *x == 0.0 {
        return None;
    }
    let y = x / 2.0;
    if *x > 0.0 && y <= 0.0 {
        return None;
    }
    if *x < 0.0 && y >= 0.0 {
        return None;
    }
    return Some(y);
}

pub fn gen_i32(size: usize, seed: u64) -> i32 {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    rng.gen::<i32>() % size as i32
}

pub fn shrink_i32(x: &i32) -> Option<i32> {
    if *x == 0 {
        return None;
    }
    let y = x / 2;
    if *x > 0 && y <= 0 {
        return None;
    }
    if *x < 0 && y >= 0 {
        return None;
    }
    return Some(y);
}

pub fn gen_i64(size: usize, seed: u64) -> i64 {
    rng(seed).gen::<i64>() % size as i64
}

// halving towards strategy.
pub fn shrink_i64(x: &i64) -> Option<i64> {
    if *x == 0 {
        return None;
    }
    let y = x / 2;
    if *x > 0 && y <= 0 {
        return None;
    }
    if *x < 0 && y >= 0 {
        return None;
    }
    return Some(y);
}

pub fn gen_u64(size: usize, seed: u64) -> u64 {
    rng(seed).gen::<u64>() % size as u64
}

// halving strategy.
pub fn shrink_u64(x: &u64) -> Option<u64> {
    if *x == 0 {
        return None;
    }
    let y = x / 2;
    return Some(y);
}

pub fn gen_vec<A>(size: usize, seed: u64, gen_element: fn(usize, u64) -> A) -> Vec<A> {
    let length = rng(seed).gen::<usize>() % size;
    (0..length)
        .into_iter()
        .map(|_| gen_element(size, seed))
        .collect()
}

pub fn gen_vec_even_i64(size: usize, seed: u64) -> Vec<i64> {
    let length = rng(seed).gen::<usize>() % size;
    (0..length)
        .into_iter()
        .map(|_| gen_i64(size, seed))
        .filter(|x| x % 2 == 0)
        .collect()
}

pub fn gen_vec_i64(size: usize, seed: u64) -> Vec<i64> {
    gen_vec(size, seed, gen_i64)
}

// basically next on an Iterator.
pub fn shrink_vec_i64(vec: &Vec<i64>) -> Option<Vec<i64>> {
    let len = vec.len();
    // actually, empty or does not cycle ...
    if vec.is_empty() {
        None
    } else {
        Some(
            vec.into_iter()
                .take(len - 1)
                .flat_map(|x| shrink_i64(x))
                .collect(),
        )
    }
}

pub fn prop_all_even(vec: &Vec<i64>) -> bool {
    if vec.is_empty() {
        return true;
    }
    vec.iter().all(|x| x % 2 == 0)
}

pub fn collatz(number: u64) -> u64 {
    if number <= 0 {
        return 1;
    }
    let mut n = number;
    while n != 1 {
        if n % 2 == 0 {
            n = n / 2;
        } else {
            n = 3 * n + 1;
        }
    }
    n
}

pub fn prop_collatz_always_one(input: &u64) -> bool {
    collatz(*input) == 1
}

pub fn abs(number: i64) -> i64 {
    if number < 0 {
        return -number;
    }
    number
}

pub fn prop_abs_always_positive(input: &i64) -> bool {
    abs(*input) >= 0
}

pub fn report<A: Debug>(witness: A, shrinks: Vec<A>) {
    // needs document styled pretty printing for data.
    println!("=== Outcome ===\n{:#?}", witness);
    if !shrinks.is_empty() {
        println!("=== Shrinks ===");
    }
    for shrink in shrinks {
        println!("{:#?}", shrink);
    }
}

pub fn oneof<A: Clone>(options: &[fn(usize) -> A], size: usize, seed: u64) -> A {
    let equally_weighted_options: Vec<_> = options.into_iter().map(|x| (1, *x)).collect();
    frequency(equally_weighted_options.as_slice(), size, seed)
}

pub fn frequency<A: Clone>(
    weighted_options: &[(usize, fn(usize) -> A)],
    size: usize,
    seed: u64,
) -> A {
    assert!(!weighted_options.is_empty());
    assert!(!weighted_options.iter().all(|(w, _)| w == &0));
    assert!(!weighted_options.iter().any(|(w, _)| w < &0));
    let total: usize = weighted_options.iter().map(|(w, _)| w).sum();
    let mut choice = rng(seed).gen::<usize>() % total + 1;
    for (weight, option) in weighted_options {
        if choice <= *weight {
            return option(size);
        }
        choice -= weight;
    }
    std::unreachable!()
}

#[derive(Debug, Clone)]
pub enum Color {
    Red,
    Green,
    Blue,
    Brown,
}

impl Color {
    pub fn is_brown(&self) -> bool {
        match self {
            Color::Brown => true,
            _ => false,
        }
    }
}

pub fn gen_color(size: usize, seed: u64) -> Color {
    oneof(
        &[
            |_| Color::Red,
            |_| Color::Green,
            |_| Color::Blue,
            |_| Color::Brown,
        ],
        size,
        seed,
    )
}

pub fn gen_color_non_brown(size: usize, seed: u64) -> Color {
    oneof(
        &[|_| Color::Red, |_| Color::Green, |_| Color::Blue],
        size,
        seed,
    )
}

pub fn shrink_color(_color: &Color) -> Option<Color> {
    None
}

pub fn prop_color_is_never_brown(color: &Color) -> bool {
    !color.is_brown()
}

#[derive(Debug, Clone)]
pub struct Coordinate {
    pub latitude: f64,
    pub longitude: f64,
}

// an example where we ignore `size'
pub fn gen_coordinate(_size: usize, seed: u64) -> Coordinate {
    let mut rng = rng(seed);
    Coordinate {
        // to play, we pretend latitude is well-behaved.
        latitude: rng.gen_range(-90.0..=90.0),
        // but that the longitude extends it's bounds somewhat.
        longitude: rng.gen_range(-1000.0..=1000.0),
    }
}

pub fn shrink_coordinate(coordinate: &Coordinate) -> Option<Coordinate> {
    let latitude = shrink_f64(&coordinate.latitude)?;
    let longitude = shrink_f64(&coordinate.longitude)?;
    Some(Coordinate {
        latitude,
        longitude,
    })
}

pub fn wrap_longitude(longitude: f64) -> f64 {
    if longitude > 180.0 {
        let wrap = ((longitude + 180.0) % 360.0) - 180.0;
        if wrap == -180.0 {
            return 180.0;
        }
        return wrap;
    }
    if longitude <= -180.0 {
        return ((longitude - 180.0) % 360.0) + 180.0;
    }
    longitude
}

pub fn wrap_coordinate(coordinate: Coordinate) -> Coordinate {
    Coordinate {
        latitude: coordinate.latitude,
        longitude: coordinate.longitude % 180.0,
    }
}

pub fn prop_wrap_longitude_always_in_bounds(coordinate: &Coordinate) -> bool {
    let wrapped = wrap_coordinate(coordinate.clone());
    wrapped.longitude <= 180.0 && wrapped.longitude > -180.0
}

pub fn sample<A>(gen: fn(usize, u64) -> A, size: usize, count: usize, seed: u64) -> Vec<A> {
    let mut buffer = Vec::with_capacity(count);
    for _ in 0..count {
        buffer.push(gen(size, seed));
    }
    buffer
}

pub struct Gen<A> {
    gen: Box<dyn Fn(usize, u64) -> A>,
    shrink: Box<dyn Fn(&A) -> Option<A>>,
}

impl<A: 'static> Gen<A> {
    pub fn new<G, S>(gen: G, shrink: S) -> Self
    where
        G: Fn(usize, u64) -> A + 'static,
        S: Fn(&A) -> Option<A> + 'static,
    {
        Gen {
            gen: Box::new(gen),
            shrink: Box::new(shrink),
        }
    }

    pub fn filter<P>(self, predicate: P) -> Self
    where
        P: Fn(&A) -> bool + 'static,
    {
        let gen = move |size, seed| {
            for _ in 0..100 {
                let generated = (self.gen)(size, seed);
                if predicate(&generated) {
                    return generated;
                }
            }
            todo!("generator discard");
        };
        Gen::new(gen, self.shrink)
    }
}

pub fn qc<A>(check: fn(&A) -> bool, gen: Gen<A>, size: usize, seed: u64, runs: usize)
where
    A: Debug + Clone + 'static,
{
    for _ in 0..runs {
        // generate.
        let input = (gen.gen)(size, seed);
        // check.
        if check(&input) {
            continue;
        }
        // shrink.
        println!("FAIL: shrinking");
        let mut trail = Vec::new();
        let smaller = (gen.shrink)(&input);
        if smaller.is_none() {
            // shrink did not produce a value.
            report(input, vec![]);
            assert!(false);
        }
        trail.push(smaller.clone().unwrap());
        let mut smallest = smaller;
        while let Some(nu) = (gen.shrink)(smallest.as_ref().unwrap()) {
            trail.push(nu.clone());
            if !check(&nu) {
                smallest = Some(nu.clone());
            }
        }
        report(smallest.unwrap(), trail);
        assert!(false);
    }
}

pub struct Qc<A> {
    runs: usize,
    size: usize, // optional size?
    seed: u64,   // one day possibly a slice of bytes.
    gen: Gen<A>,
}

impl<A: Debug + Clone + 'static> Qc<A> {
    pub fn new(gen: Gen<A>) -> Self {
        Qc {
            runs: 100,
            size: 100, // random as well?
            seed: rand::random(),
            gen: gen,
        }
    }

    pub fn with_runs(self, runs: usize) -> Self {
        Qc { runs, ..self }
    }

    pub fn with_size(self, size: usize) -> Self {
        Qc { size, ..self }
    }

    pub fn with_seed(self, seed: u64) -> Self {
        Qc { seed, ..self }
    }

    pub fn check(self, property: fn(&A) -> bool) {
        qc(property, self.gen, self.size, self.seed, self.runs)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    //#[test]
    //fn test_sample() {
    //    println!("{:#?}", sample(gen_coordinate, f64::MAX as usize, 10));
    //    assert!(false);
    //}

    #[test]
    fn playground_qc() {
        qc(
            prop_all_even,
            Gen::new(gen_vec_even_i64, shrink_vec_i64),
            100,
            rand::random(),
            100,
        );
        qc(
            prop_collatz_always_one,
            Gen::new(gen_u64, shrink_u64),
            1000,
            // if we go as large as the full space for u64, we will
            // run into panics on multiply, which is good! but not
            // the point of this playground and prototype.
            //u64::MAX as usize,
            rand::random(),
            100,
        );
        qc(
            prop_abs_always_positive,
            Gen::new(gen_i64, shrink_i64),
            100,
            rand::random(),
            100,
        );
    }

    #[test]
    fn playground_builder() {
        Qc::new(Gen::new(gen_vec_even_i64, shrink_vec_i64)).check(prop_all_even);

        Qc::new(Gen::new(gen_u64, shrink_u64))
            .with_size(1000)
            .check(prop_collatz_always_one);

        Qc::new(Gen::new(gen_i64, shrink_i64))
            .with_size(i64::MAX as usize)
            .check(prop_abs_always_positive);

        Qc::new(Gen::new(gen_color_non_brown, shrink_color))
            .with_size(i64::MAX as usize)
            .check(prop_color_is_never_brown);

        Qc::new(Gen::new(gen_coordinate, shrink_coordinate))
            .with_size(f64::MAX as usize)
            .check(prop_wrap_longitude_always_in_bounds);

        let none = None;
        Qc::new(Gen::new(gen_coordinate, move |_| none.clone()))
            .with_size(f64::MAX as usize)
            .check(prop_wrap_longitude_always_in_bounds);
    }

    #[test]
    fn playground_filter() {
        let gen_even_i64 = Gen::new(gen_i64, shrink_i64).filter(|x| x % 2 == 0);
        Qc::new(gen_even_i64).with_seed(5).check(|x| x % 2 == 0);
    }

    #[test]
    fn playground_shrinks_always_smaller() {
        let gen = Gen::new(gen_u64, shrink_u64);
        let mut last_element = Some((gen.gen)(u64::MAX as usize, rand::random()));
        while let Some(element) = (gen.shrink)(last_element.as_ref().unwrap()) {
            match last_element {
                None => last_element = Some(element),
                Some(ref last) => {
                    assert!(element < *last);
                    last_element = Some(element);
                }
            }
        }
    }

    #[test]
    fn playground_seed_gen_idempotent() {
        let gen = Gen::new(gen_i32, shrink_i32);
        let size = i32::MAX as usize;
        let seed = 128;
        let fst = (gen.gen)(size, seed);
        let snd = (gen.gen)(size, seed);
        assert_eq!(fst, snd);
    }
}
