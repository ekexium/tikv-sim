use std::sync::atomic;
use std::sync::atomic::AtomicU64;

pub type Time = u64;

pub const NANOSECOND: Time = 1;
pub const MICROSECOND: Time = 1_000 * NANOSECOND;
pub const MILLISECOND: Time = 1_000 * MICROSECOND;
pub const SECOND: Time = 1_000 * MILLISECOND;

pub trait TimeTrait {
    fn pretty_print(&self) -> String;
}

impl TimeTrait for Time {
    fn pretty_print(&self) -> String {
        if *self < MICROSECOND {
            format!("{} ns", self)
        } else if *self < MILLISECOND {
            format!("{:.2} us", *self as f64 / MICROSECOND as f64)
        } else if *self < SECOND {
            format!("{:.2} ms", *self as f64 / MILLISECOND as f64)
        } else {
            format!("{:.2} s", *self as f64 / SECOND as f64)
        }
    }
}

pub static CURRENT_TIME: AtomicU64 = AtomicU64::new(0);

pub fn now() -> Time {
    CURRENT_TIME.load(atomic::Ordering::SeqCst)
}
