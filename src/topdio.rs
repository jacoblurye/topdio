use std::{
    cmp::Ordering,
    ffi::OsString,
    thread::sleep,
    time::{Duration, Instant},
};

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use sysinfo::{Process, ProcessRefreshKind, ProcessesToUpdate, RefreshKind, System};

#[derive(Debug, Clone, PartialEq)]
pub struct ProcessInfo {
    pub pid: u32,
    pub name: OsString,
    pub cpu_usage: f32,
}

impl ProcessInfo {
    pub fn from_process(process: &Process) -> ProcessInfo {
        ProcessInfo {
            pid: process.pid().as_u32(),
            name: process.name().to_os_string(),
            cpu_usage: process.cpu_usage(),
        }
    }
}

fn order_by_cpu(p1: &ProcessInfo, p2: &ProcessInfo) -> Ordering {
    if p1.cpu_usage < p2.cpu_usage {
        Ordering::Less
    } else if p1.cpu_usage == p2.cpu_usage {
        Ordering::Equal
    } else {
        Ordering::Greater
    }
}

/// A message from [`Topdio`] to its subscribers.
#[derive(Debug, Clone, PartialEq)]
pub enum TopdioMessage {
    /// Current system info statistics. If send by [`Topdio`], the processes are sorted
    /// by descending CPU usage.
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
            .filter(|p| p.pid != std::process::id())
            .map(|p| p.to_owned())
            .collect();
        // Sort the processes by descending CPU usage.
        processes.sort_by(|p1, p2| order_by_cpu(p1, p2).reverse());

        TopdioMessage::Stats { processes }
    }
}

pub trait TopdioSubscriber {
    fn handle(&mut self, message: &TopdioMessage) -> Result<()>;
}

/// Gathers system info and broadcasts it to a set of subscribers.
pub struct Topdio {
    refresh_rate: Duration,
    subscribers: Vec<Box<dyn TopdioSubscriber>>,
}

impl Topdio {
    pub fn new(subscribers: Vec<Box<dyn TopdioSubscriber>>, refresh_rate: f32) -> Topdio {
        Topdio {
            refresh_rate: Duration::from_secs_f32(refresh_rate),
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

    /// Continuously gather system stats and publishing them to subscribers.
    pub fn run(&mut self) -> Result<()> {
        let mut system = System::new_with_specifics(
            RefreshKind::nothing().with_processes(ProcessRefreshKind::everything()),
        );

        // Initial process data won't be accurate - refresh a couple times before sending.
        for _ in 0..2 {
            system.refresh_processes(ProcessesToUpdate::All, true);
            sleep(Duration::from_millis(200));
        }

        let mut last_tick = Instant::now();
        loop {
            system.refresh_processes(ProcessesToUpdate::All, true);

            let processes: Vec<ProcessInfo> = system
                .processes()
                .iter()
                .map(|(_, p)| ProcessInfo::from_process(p))
                .collect();
            let message = TopdioMessage::stats(&processes);
            self.broadcast(&message)?;

            let timeout = self
                .refresh_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if Self::poll_quit(timeout)? {
                break;
            }
            if last_tick.elapsed() >= self.refresh_rate {
                last_tick = Instant::now();
            }
        }

        self.broadcast(&TopdioMessage::Stop)?;

        Ok(())
    }

    fn poll_quit(timeout: Duration) -> Result<bool> {
        if crossterm::event::poll(timeout)? {
            if let Event::Key(key_event) = crossterm::event::read()? {
                match key_event {
                    // Quit on 'q' or ctrl-c.
                    KeyEvent {
                        code: KeyCode::Char('q'),
                        ..
                    }
                    | KeyEvent {
                        code: KeyCode::Char('c'),
                        modifiers: KeyModifiers::CONTROL,
                        ..
                    } => return Ok(true),
                    _ => return Ok(false),
                }
            }
        }
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_constructor() {
        let stats = TopdioMessage::stats(&vec![
            ProcessInfo {
                pid: 1,
                name: "p1".into(),
                cpu_usage: 2.,
            },
            ProcessInfo {
                pid: std::process::id(),
                name: "p2".into(),
                cpu_usage: 1.,
            },
            ProcessInfo {
                pid: 1,
                name: "p3".into(),
                cpu_usage: 3.,
            },
        ]);

        assert_eq!(
            stats,
            TopdioMessage::Stats {
                processes: vec![
                    ProcessInfo {
                        pid: 1,
                        name: "p3".into(),
                        cpu_usage: 3.,
                    },
                    ProcessInfo {
                        pid: 1,
                        name: "p1".into(),
                        cpu_usage: 2.,
                    },
                ]
            }
        )
    }
}
