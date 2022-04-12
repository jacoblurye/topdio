use std::{
    cmp::Ordering,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossterm::{
    event::{Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use sysinfo::{PidExt, Process, ProcessExt, System, SystemExt};

/// Info about a system process.
#[derive(Debug, Clone, PartialEq)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: String,
    pub cpu_usage: f32,
}

impl ProcessInfo {
    /// Build a [`ProcessInfo`] from a [`sysinfo::Process`].
    pub fn new(process: &Process) -> ProcessInfo {
        ProcessInfo {
            pid: process.pid().as_u32(),
            name: process.name().to_string(),
            cpu_usage: process.cpu_usage(),
        }
    }
}

/// A message from [`Topdio`] to its subscribers.
#[derive(Debug, Clone, PartialEq)]
pub enum TopdioMessage {
    /// Current system info statistics. If receiving this message from a [`Topdio`],
    /// subscribers can rely on the invariant that `processes` will exclude the currently
    /// running process and be sorted by decreasing CPU usage.
    Stats { processes: Vec<ProcessInfo> },
    /// Subscriber should clean up and shut down if they receive this variant.
    Stop,
}

impl TopdioMessage {
    /// Create a [`TopdioMessage::Stats`] from the given list of processes, filtering out
    /// the currently running process and sorting processes by descreasing CPU usage.
    fn stats(processes: &[ProcessInfo]) -> TopdioMessage {
        // Exclude the currently running process.
        let mut processes: Vec<ProcessInfo> = processes
            .iter()
            .map(|p| p.to_owned())
            .filter(|p| p.pid != std::process::id())
            .collect();
        // Sort the processes by descending CPU usage.
        processes.sort_by(|p1, p2| {
            if p1.cpu_usage >= p2.cpu_usage {
                Ordering::Less
            } else {
                Ordering::Greater
            }
        });

        TopdioMessage::Stats { processes }
    }
}

pub trait TopdioSubscriber {
    fn handle(&mut self, message: &TopdioMessage) -> Result<()>;
}

pub trait TopdioQuitter {
    fn poll_quit(&self, timeout: Duration) -> Result<bool>;
}

/// Polls for ctrl-c or q input from the user using [`crossterm::event::poll`].
pub struct CrosstermQuitter {}

impl CrosstermQuitter {
    pub fn new() -> Result<CrosstermQuitter> {
        enable_raw_mode()?;
        Ok(CrosstermQuitter {})
    }
}

impl Drop for CrosstermQuitter {
    fn drop(&mut self) {
        disable_raw_mode().unwrap();
    }
}

const Q: KeyEvent = KeyEvent {
    code: KeyCode::Char('q'),
    modifiers: KeyModifiers::NONE,
};
const CTRL_C: KeyEvent = KeyEvent {
    code: KeyCode::Char('c'),
    modifiers: KeyModifiers::CONTROL,
};

impl TopdioQuitter for CrosstermQuitter {
    fn poll_quit(&self, timeout: Duration) -> Result<bool> {
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key_event) = crossterm::event::read()? {
                if key_event == Q || key_event == CTRL_C {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }
}

/// Gathers system info and broadcasts it to a set of subscribers.
pub struct Topdio {
    refresh_rate: Duration,
    subscribers: Vec<Box<dyn TopdioSubscriber>>,
}

impl Topdio {
    pub fn new(subscribers: Vec<Box<dyn TopdioSubscriber>>, refresh_rate: u64) -> Topdio {
        Topdio {
            refresh_rate: Duration::from_millis(refresh_rate),
            subscribers,
        }
    }

    fn broadcast(&mut self, message: &TopdioMessage) -> Result<()> {
        let mut error = None;
        for subscriber in self.subscribers.iter_mut() {
            if let Err(e) = subscriber.handle(message) {
                error = Some(e);
                break;
            };
        }

        if let Some(e) = error {
            for subscriber in self.subscribers.iter_mut() {
                let _ = subscriber.handle(&TopdioMessage::Stop);
            }
            return Err(e);
        }

        Ok(())
    }

    /// Run the core system info loop, gathering system stats and publishing them to
    /// subscribers until the `quitter` says to stop.
    pub fn run<T>(&mut self, quitter: &T) -> Result<()>
    where
        T: TopdioQuitter,
    {
        let mut system = System::new_all();
        let mut last_tick = Instant::now();
        loop {
            system.refresh_all();

            let processes: Vec<ProcessInfo> = system
                .processes()
                .iter()
                .map(|(_, p)| ProcessInfo::new(p))
                .collect();
            let message = TopdioMessage::stats(&processes);
            self.broadcast(&message)?;

            let timeout = self
                .refresh_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));
            if quitter.poll_quit(timeout)? {
                break;
            }
            if last_tick.elapsed() >= self.refresh_rate {
                last_tick = Instant::now();
            }
        }

        self.broadcast(&TopdioMessage::Stop)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use anyhow::bail;

    use super::*;

    #[test]
    fn test_stats_constructor() {
        let stats = TopdioMessage::stats(&vec![
            ProcessInfo {
                pid: 1,
                name: "p1".to_string(),
                cpu_usage: 2.,
            },
            ProcessInfo {
                pid: std::process::id(),
                name: "p2".to_string(),
                cpu_usage: 1.,
            },
            ProcessInfo {
                pid: 1,
                name: "p3".to_string(),
                cpu_usage: 3.,
            },
        ]);

        assert_eq!(
            stats,
            TopdioMessage::Stats {
                processes: vec![
                    ProcessInfo {
                        pid: 1,
                        name: "p3".to_string(),
                        cpu_usage: 3.,
                    },
                    ProcessInfo {
                        pid: 1,
                        name: "p1".to_string(),
                        cpu_usage: 2.,
                    },
                ]
            }
        )
    }

    struct TestSubscriber {
        error_after: Option<usize>,
        messages: Rc<RefCell<Vec<TopdioMessage>>>,
    }

    impl TopdioSubscriber for TestSubscriber {
        fn handle(&mut self, message: &TopdioMessage) -> Result<()> {
            let mut messages = self.messages.borrow_mut();
            messages.push(message.clone());
            if let Some(error_after) = self.error_after {
                if messages.len() > error_after {
                    bail!("erroring!")
                }
            }
            Ok(())
        }
    }

    struct TestQuitter {
        quit_after: Option<usize>,
        calls: RefCell<Vec<Duration>>,
    }

    impl TestQuitter {
        fn new(quit_after: Option<usize>) -> TestQuitter {
            TestQuitter {
                quit_after,
                calls: RefCell::new(vec![]),
            }
        }
    }

    impl TopdioQuitter for TestQuitter {
        fn poll_quit(&self, timeout: Duration) -> Result<bool> {
            let mut calls = self.calls.borrow_mut();
            calls.push(timeout);
            if let Some(quit_after) = self.quit_after {
                if calls.len() > quit_after {
                    return Ok(true);
                }
            }
            Ok(false)
        }
    }

    #[test]
    fn test_loop_and_quit() {
        let messages = Rc::new(RefCell::new(vec![]));
        let subscriber = TestSubscriber {
            error_after: None,
            messages: messages.clone(),
        };
        let quitter = TestQuitter::new(Some(1));
        let mut topdio = Topdio::new(vec![Box::new(subscriber)], 100);
        topdio.run(&quitter).unwrap();
        assert_eq!(quitter.calls.borrow().len(), 2 as usize);
        assert_eq!(messages.borrow().len(), 3 as usize);
        assert_eq!(messages.borrow().last(), Some(&TopdioMessage::Stop));
    }

    #[test]
    fn test_loop_and_error() {
        let messages_1 = Rc::new(RefCell::new(vec![]));
        let subscriber_1 = TestSubscriber {
            error_after: Some(1),
            messages: messages_1.clone(),
        };
        let messages_2 = Rc::new(RefCell::new(vec![]));
        let subscriber_2 = TestSubscriber {
            error_after: None,
            messages: messages_2.clone(),
        };
        let quitter = TestQuitter::new(None);
        let mut topdio = Topdio::new(vec![Box::new(subscriber_1), Box::new(subscriber_2)], 100);
        let err = topdio.run(&quitter).unwrap_err();
        assert_eq!(err.to_string(), "erroring!");
        assert_eq!(messages_1.borrow().len(), 3);
        assert_eq!(messages_1.borrow().last(), Some(&TopdioMessage::Stop));
        assert_eq!(messages_2.borrow().len(), 2);
        assert_eq!(messages_2.borrow().last(), Some(&TopdioMessage::Stop));
    }
}
