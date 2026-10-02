/// Conditionally print a formatted message to stdout.
///
/// Evaluates `$cond` at runtime; if `true`, calls [`println!`] with the
/// remaining arguments unchanged.  When `$cond` is a compile-time constant
/// `false`, the body is still compiled but the optimizer will eliminate it.
///
/// # Example
///
/// ```rust
/// use fast_umap::print_if;
/// const VERBOSE: bool = true;
/// let epoch = 1;
/// let loss = 0.25;
/// print_if!(VERBOSE, "step {} loss = {:.4}", epoch, loss);
/// ```
#[macro_export]
macro_rules! print_if {
    ($cond:expr, $($arg:tt)*) => {
        if $cond {
            println!($($arg)*);
        }
    };
}
