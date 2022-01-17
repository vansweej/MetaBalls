#[macro_export]
macro_rules! compose {
    ( $last:expr ) => { $last };
    ( $head:expr, $($tail:expr), +) => {
        compose_two($head, compose!($($tail),+))
    };
}

pub fn compose_two<A, B, C, G, F>(f: F, g: G) -> impl Fn(A) -> C
    where
        F: Fn(A) -> B,
        G: Fn(B) -> C,
{
    move |x| g(f(x))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose1() {
        let add = |x| x + 2;
        let multiply = |x| x * 2;
        let divide = |x| x / 2;
        let intermediate = compose!(add, multiply, divide);

        let subtract = |x| x - 2;
        let finally = compose!(intermediate, subtract);

        println!("Result is {}", finally(10));
    }

    fn add(x: i32) -> i32 {
        x + 2
    }

    fn multiply(x: i32) -> i32 {
        x * 2
    }

    fn divide(x: i32) -> i32 {
        x / 2
    }


    #[test]
    fn test_compose2() {
        let intermediate = compose!(add, multiply, divide);

        let subtract = |x| x - 2;
        let finally = compose!(intermediate, subtract);

        println!("Result is {}", finally(10));
    }
}