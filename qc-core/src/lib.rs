use std::fmt::Debug;

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

pub fn sample<A>(gen: fn(usize) -> A, size: usize, count: usize) -> Vec<A> {
    let mut buffer = Vec::with_capacity(count);
    for _ in 0..count {
        buffer.push(gen(size));
    }
    buffer
}

// remove need for Clone bound.
// shrinking needs to be made optional?
// probably should be a macro, to simplify passing, etc.
// but doesn't have to be.
pub fn qc<A: Debug + Clone>(
    check: fn(&A) -> bool,
    gen: fn(usize) -> A,
    // actually should be an Iterator.
    // OR could create the Iterator from the pure function.
    shrink: fn(A) -> Option<A>,
    size: usize,
    runs: usize,
) {
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
            if check(&nu.clone().unwrap()) {
                // deadend. can stop shrinking.
                break;
            }
            if nu.is_none() {
                // can't shrink anymore.
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
    size: usize, // TODO: ought to be optional.
    gen: Option<fn(usize) -> A>,
    shrink: Option<fn(A) -> Option<A>>,
}

impl<A: Debug + Clone> Qc<A> {
    pub fn new() -> Self {
        Qc {
            runs: 100,
            size: 100,
            gen: None,
            shrink: None,
        }
    }

    pub fn with_gen(self, gen: fn(usize) -> A) -> Self {
        Qc {
            gen: Some(gen),
            ..self
        }
    }

    pub fn with_shrink(self, shrink: fn(A) -> Option<A>) -> Self {
        Qc {
            shrink: Some(shrink),
            ..self
        }
    }

    pub fn with_runs(self, runs: usize) -> Self {
        Qc { runs, ..self }
    }

    pub fn with_size(self, size: usize) -> Self {
        Qc { size, ..self }
    }

    pub fn check(self, property: fn(&A) -> bool) {
        assert!(self.gen.is_some());
        assert!(self.shrink.is_some());
        qc(
            property,
            self.gen.unwrap(),
            self.shrink.unwrap(),
            self.size,
            self.runs,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    //#[test]
    //fn test_sample() {
    //    println!("{:#?}", sample(gen_color, 3, 100));
    //    assert!(false);
    //}

    #[test]
    fn playground() {
        qc(prop_all_even, gen_vec_even_i64, shrink_vec_i64, 100, 100);
        qc(
            prop_collatz_always_one,
            gen_u64,
            shrink_u64,
            1000,
            // if we go as large as the full space for u64, we will
            // run into panics on multiply, which is good! but not
            // the point of this playground and prototype.
            //u64::MAX as usize,
            100,
        );
        qc(prop_abs_always_positive, gen_i64, shrink_i64, 100, 100);

        Qc::new()
            .with_gen(gen_vec_even_i64)
            .with_shrink(shrink_vec_i64)
            .check(prop_all_even);

        Qc::new()
            .with_gen(gen_u64)
            .with_shrink(shrink_u64)
            .with_size(1000)
            .check(prop_collatz_always_one);

        Qc::new()
            .with_gen(gen_i64)
            .with_shrink(shrink_i64)
            .with_size(i64::MAX as usize)
            .check(prop_abs_always_positive);

        Qc::new()
            .with_gen(gen_color_non_brown)
            .with_shrink(shrink_color)
            .with_size(i64::MAX as usize)
            .check(prop_color_is_never_brown);
    }
}
