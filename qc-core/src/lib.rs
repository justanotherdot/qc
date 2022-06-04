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

pub fn gen_vec_i64(size: usize) -> Vec<i64> {
    let length = rand::random::<usize>() % size;
    (0..length)
        .into_iter()
        .map(|_| gen_i64(size))
        .filter(|x| x % 2 == 0)
        .collect()
}

// basically next on an Iterator.
pub fn shrink_vec_i64(vec: Vec<i64>) -> Option<Vec<i64>> {
    let len = vec.len();
    // actually, empty or does not cycle ...
    if vec.is_empty() {
        None
    } else {
        // TODO: for now, shrink the len, but later, also shrink the elements.
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

pub fn report<A: Debug>(rounds: usize, witness: A) {
    // needs document styled pretty printing for data.
    println!(
        "found smallest shrink after {} rounds\n  {:#?}",
        rounds, witness
    );
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

#[cfg(test)]
mod test {
    use super::*;

    //#[test]
    //fn sample() {
    //    for _ in 0..100 {
    //        let list = gen_vec_i64(3);
    //        dbg!(&list);
    //    }
    //    assert!(false);
    //}

    //#[test]
    //fn prop_all_even_works() {
    //    assert!(prop_all_even(&Vec::new()));
    //    assert!(prop_all_even(&[2].into_iter().collect()));
    //    assert!(prop_all_even(&[4, 8].into_iter().collect()));
    //}

    //#[test]
    //fn prop_collatz_always_one_works() {
    //    for n in 0..100 {
    //        assert!(prop_collatz_always_one(&n));
    //    }
    //}

    #[test]
    fn playground() {
        qc(prop_all_even, gen_vec_i64, shrink_vec_i64, 100, 100);
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
    }
}
