use std::fmt::Debug;

pub fn gen_f64(size: usize) -> f64 {
    rand::random::<f64>() % size as f64
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

pub fn gen_i64(size: usize) -> i64 {
    rand::random::<i64>() % size as i64
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

pub fn gen_u64(size: usize) -> u64 {
    rand::random::<u64>() % size as u64
}

// halving strategy.
pub fn shrink_u64(x: &u64) -> Option<u64> {
    if *x == 0 {
        return None;
    }
    let y = x / 2;
    return Some(y);
}

pub fn gen_vec<A>(size: usize, gen_element: fn(usize) -> A) -> Vec<A> {
    let length = rand::random::<usize>() % size;
    (0..length).into_iter().map(|_| gen_element(size)).collect()
}

pub fn gen_vec_even_i64(size: usize) -> Vec<i64> {
    let length = rand::random::<usize>() % size;
    (0..length)
        .into_iter()
        .map(|_| gen_i64(size))
        .filter(|x| x % 2 == 0)
        .collect()
}

pub fn gen_vec_i64(size: usize) -> Vec<i64> {
    gen_vec(size, gen_i64)
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

pub fn oneof<A: Clone>(options: &[fn(usize) -> A], size: usize) -> A {
    let equally_weighted_options: Vec<_> = options.into_iter().map(|x| (1, *x)).collect();
    frequency(equally_weighted_options.as_slice(), size)
}

pub fn frequency<A: Clone>(weighted_options: &[(usize, fn(usize) -> A)], size: usize) -> A {
    assert!(!weighted_options.is_empty());
    assert!(!weighted_options.iter().all(|(w, _)| w == &0));
    assert!(!weighted_options.iter().any(|(w, _)| w < &0));
    let total: usize = weighted_options.iter().map(|(w, _)| w).sum();
    let mut choice = rand::random::<usize>() % total + 1;
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

pub fn gen_color(size: usize) -> Color {
    oneof(
        &[
            |_| Color::Red,
            |_| Color::Green,
            |_| Color::Blue,
            |_| Color::Brown,
        ],
        size,
    )
}

pub fn gen_color_non_brown(size: usize) -> Color {
    oneof(&[|_| Color::Red, |_| Color::Green, |_| Color::Blue], size)
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
pub fn gen_coordinate(_size: usize) -> Coordinate {
    use rand::Rng;
    let mut rng = rand::thread_rng();
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

pub fn sample<A>(gen: fn(usize) -> A, size: usize, count: usize) -> Vec<A> {
    let mut buffer = Vec::with_capacity(count);
    for _ in 0..count {
        buffer.push(gen(size));
    }
    buffer
}

pub struct Gen<A> {
    gen: Box<dyn Fn(usize) -> A>,
    shrink: Box<dyn Fn(&A) -> Option<A>>,
}

impl<A: 'static> Gen<A> {
    pub fn new<G, S>(gen: G, shrink: S) -> Self
    where
        G: Fn(usize) -> A + 'static,
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
        let gen = move |size| {
            for _ in 0..100 {
                let generated = (self.gen)(size);
                if predicate(&generated) {
                    return generated;
                }
            }
            todo!("generator discard");
        };
        Gen::new(gen, self.shrink)
    }
}

pub fn qc<A>(check: fn(&A) -> bool, gen: Gen<A>, size: usize, runs: usize)
where
    A: Debug + Clone + 'static,
{
    for _ in 0..runs {
        // generate.
        let input = (gen.gen)(size);
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
    gen: Gen<A>,
}

impl<A: Debug + Clone + 'static> Qc<A> {
    pub fn new(gen: Gen<A>) -> Self {
        Qc {
            runs: 100,
            size: 100,
            gen: gen,
        }
    }

    pub fn with_runs(self, runs: usize) -> Self {
        Qc { runs, ..self }
    }

    pub fn with_size(self, size: usize) -> Self {
        Qc { size, ..self }
    }

    pub fn check(self, property: fn(&A) -> bool) {
        qc(property, self.gen, self.size, self.runs)
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
            100,
        );
        qc(
            prop_abs_always_positive,
            Gen::new(gen_i64, shrink_i64),
            100,
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
        Qc::new(gen_even_i64).check(|x| x % 2 == 0);
    }

    #[test]
    fn playground_gen_iter() {
        let gen = Gen::new(gen_u64, shrink_u64);
        let mut last_element = Some((gen.gen)(u64::MAX as usize));
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
}
