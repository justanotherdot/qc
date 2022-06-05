use std::fmt::Debug;

pub fn gen_f64(size: usize) -> f64 {
    rand::random::<f64>() % size as f64
}

// towards zero strategy.
pub fn shrink_f64(number: f64) -> Option<f64> {
    if number > 0.0 {
        let nu = number / 2.0;
        if nu < 0.0 {
            return Some(0.0);
        }
        return Some(nu);
    } else if number < 0.0 {
        let nu = number * 2.0;
        if nu > 0.0 {
            return Some(0.0);
        }
        return Some(nu);
    } else {
        None
    }
}

pub fn gen_i64(size: usize) -> i64 {
    rand::random::<i64>() % size as i64
}

// halving strategy.
pub fn shrink_i64(number: i64) -> Option<i64> {
    Some(number / 2)
}

pub fn gen_u64(size: usize) -> u64 {
    rand::random::<u64>() % size as u64
}

// halving strategy.
pub fn shrink_u64(number: u64) -> Option<u64> {
    Some(number / 2)
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
pub fn shrink_vec_i64(vec: Vec<i64>) -> Option<Vec<i64>> {
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

// could also do an abs check for playground.
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

pub fn report<A: Debug>(rounds: usize, witness: A) {
    // needs document styled pretty printing for data.
    println!(
        "found smallest shrink after {} rounds\n  {:#?}",
        rounds, witness
    );
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

pub fn shrink_color(_color: Color) -> Option<Color> {
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

pub fn shrink_coordinate(coordinate: Coordinate) -> Option<Coordinate> {
    let latitude = shrink_f64(coordinate.latitude)?;
    let longitude = shrink_f64(coordinate.longitude)?;
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

// Needs size _and_ seed.
// also needs ability to _ignore_ size.
// unless gen is the thing that ignores shrink?
pub struct Gen<A> {
    pub gen: Box<dyn Fn(usize) -> A>,
    pub shrink: Box<dyn Fn(A) -> Option<A>>,
    //pub shrinks: Box<dyn Iterator<Item = A>>,
}

impl<A: 'static> Gen<A> {
    pub fn new<G, S>(gen: G, shrink: S) -> Self
    where
        G: Fn(usize) -> A + 'static,
        S: Fn(A) -> Option<A> + 'static,
    {
        Gen {
            gen: Box::new(gen),
            shrink: Box::new(shrink),
        }
    }

    // This is a bit funny. ideally the shrink isn't specified,
    // and you rely on the underlying shrink, and modify that.
    pub fn map<B: 'static, F, S>(self, f: F, shrink: S) -> Gen<B>
    where
        F: Fn(A) -> B + 'static,
        S: Fn(B) -> Option<B> + 'static,
    {
        let gen = move |size| f((self.gen)(size));
        Gen::new(gen, shrink)
    }
}

// remove need for Clone bound?
// shrinking needs to be made optional?
pub fn qc<A: Debug + Clone>(check: fn(&A) -> bool, gen: Gen<A>, size: usize, runs: usize) {
    let Gen { gen, shrink } = gen;
    for _ in 0..runs {
        // generate.
        let input = gen(size);
        // check.
        if check(&input) {
            continue;
        }
        // shrink.
        println!("FAIL: shrinking");
        let mut rounds = 1;
        let mut smaller = shrink(input.clone());
        if smaller.is_none() {
            // shrink did not produce a value.
            report(rounds, input);
            assert!(false);
            break;
        }
        if check(&smaller.clone().unwrap()) {
            // shrink did not produce a failure.
            // TODO: this ought to be _shrinks_ and not shrink.
            report(rounds, input);
            assert!(false);
            break;
        }
        while let Some(ref s) = smaller {
            rounds += 1;
            let nu = shrink(s.clone());
            if nu.is_none() {
                // can't shrink anymore.
                break;
            }
            if check(&nu.clone().unwrap()) {
                // deadend. can stop shrinking.
                break;
            }
            smaller = nu;
        }
        report(rounds, smaller.unwrap());
        assert!(false);
        break;
    }
}

pub struct Qc<A> {
    runs: usize,
    size: usize, // optional size?
    gen: Gen<A>,
}

impl<A: Debug + Clone> Qc<A> {
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
    pub fn playground_gen_map() {
        #[derive(Debug, Clone)]
        struct Int32(i32);
        let gen_i32 = Gen::new(|_| rand::random::<i32>(), |_| None);
        let gen_int32 = gen_i32.map(|integer| Int32(integer), |_| None);
        Qc::new(gen_int32).check(|_| false);
    }
}
