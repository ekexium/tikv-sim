/// https://learn.microsoft.com/en-us/azure/architecture/patterns/circuit-breaker
use crate::Time;

#[derive(Debug, Eq, PartialOrd, PartialEq, Ord)]
enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

pub struct CircuitBreaker {
    state: CircuitBreakerState,

    failure_counter: u64,
    // threshold from Closed to Open
    failure_threshold: u64,
    success_counter: u64,
    // threshold from HalfOpen to Closed
    success_threshold: u64,
    // wait time before from Open to HalfOpen
    timeout: Time,
    time_of_open: Time,
    half_open_tokens: u64,
    half_open_allowed_counter: u64,
}

impl CircuitBreaker {
    pub fn new(
        success_threshold: u64,
        failure_threshold: u64,
        half_open_tokens: u64,
        timeout: Time,
        now: Time,
    ) -> Self {
        CircuitBreaker {
            state: CircuitBreakerState::Closed,
            timeout,
            time_of_open: now,
            failure_counter: 0,
            failure_threshold,
            success_counter: 0,
            success_threshold,
            half_open_tokens,
            half_open_allowed_counter: 0,
        }
    }

    fn change_to_closed(&mut self) {
        self.failure_counter = 0;
        self.state = CircuitBreakerState::Closed;
    }

    fn change_to_open(&mut self, now: Time) {
        self.time_of_open = now;
        self.state = CircuitBreakerState::Open;
    }

    fn change_to_half_open(&mut self) {
        self.success_counter = 0;
        self.half_open_allowed_counter = 0;
        self.state = CircuitBreakerState::HalfOpen;
    }

    pub fn record_failure(&mut self, now: Time) {
        match self.state {
            CircuitBreakerState::Closed => {
                self.failure_counter += 1;
                if self.failure_counter >= self.failure_threshold {
                    self.change_to_open(now);
                }
            }
            CircuitBreakerState::Open => {}
            CircuitBreakerState::HalfOpen => {
                // any failure result in the breaker to open
                self.change_to_open(now);
            }
        }
    }

    pub fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::Closed => {}
            CircuitBreakerState::Open => {}
            CircuitBreakerState::HalfOpen => {
                self.success_counter += 1;
                if self.success_counter >= self.success_threshold {
                    self.change_to_closed();
                }
            }
        }
    }

    pub fn allow_request(&mut self, now: Time) -> bool {
        // checker timer
        if matches!(self.state, CircuitBreakerState::Open)
            && now - self.time_of_open >= self.timeout
        {
            self.change_to_half_open();
        }

        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => false,
            CircuitBreakerState::HalfOpen => {
                if self.half_open_allowed_counter < self.half_open_tokens {
                    self.half_open_allowed_counter += 1;
                    true
                } else {
                    false
                }
            }
        }
    }

    pub fn status_string(&self) -> String {
        match self.state {
            CircuitBreakerState::Closed => "Closed".to_string(),
            CircuitBreakerState::Open => "Open".to_string(),
            CircuitBreakerState::HalfOpen => "HalfOpen".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_closed_to_open_transition() {
        let mut cb = CircuitBreaker::new(2, 3, 1, 10, 0);

        // Record two failures, state should still be Closed
        cb.record_failure(1);
        cb.record_failure(2);
        assert_eq!(cb.state, CircuitBreakerState::Closed);

        // Record one more failure, state should transition to Open
        cb.record_failure(3);
        assert_eq!(cb.state, CircuitBreakerState::Open);
    }

    #[test]
    fn test_open_to_half_open_transition() {
        let mut cb = CircuitBreaker::new(2, 3, 1, 10, 0);
        cb.change_to_open(0);

        // Check if request is allowed before retry_timeout
        assert_eq!(cb.allow_request(5), false);
        assert_eq!(cb.state, CircuitBreakerState::Open);

        // Check if request is allowed after retry_timeout
        assert_eq!(cb.allow_request(11), true);
        assert_eq!(cb.state, CircuitBreakerState::HalfOpen);
    }

    #[test]
    fn test_half_open_behavior() {
        let mut cb = CircuitBreaker::new(2, 3, 1, 10, 0);
        cb.change_to_half_open();

        // Record a success, state should still be HalfOpen
        cb.record_success();
        assert_eq!(cb.state, CircuitBreakerState::HalfOpen);

        // Record another success, state should transition to Closed
        cb.record_success();
        assert_eq!(cb.state, CircuitBreakerState::Closed);

        // Transition back to HalfOpen and record a failure
        cb.change_to_half_open();
        cb.record_failure(15);
        assert_eq!(cb.state, CircuitBreakerState::Open);
    }

    #[test]
    fn test_half_open_request_limit() {
        let mut cb = CircuitBreaker::new(2, 3, 1, 10, 0);
        cb.change_to_half_open();

        // Only one request should be allowed in HalfOpen state
        assert_eq!(cb.allow_request(12), true);
        assert_eq!(cb.allow_request(13), false);
    }
}
