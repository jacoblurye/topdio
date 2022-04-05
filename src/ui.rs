use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{stdout, Stdout};
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::Color,
    style::Style,
    widgets::{Block, Borders, Row, Table},
    Terminal,
};

use crate::topdio::{ProcessInfo, TopdioMessage, TopdioSubscriber};

const COLORS: [Color; 6] = [
    Color::LightBlue,
    Color::LightCyan,
    Color::LightGreen,
    Color::LightMagenta,
    Color::LightRed,
    Color::LightYellow,
];

fn get_process_color(pid: u32) -> Color {
    COLORS[(pid as usize) % COLORS.len()]
}

pub struct UI {
    terminal: Terminal<CrosstermBackend<Stdout>>,
}

impl UI {
    pub fn new() -> Result<UI> {
        enable_raw_mode()?;
        let mut stdout = stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let terminal = Terminal::new(backend)?;

        Ok(UI { terminal })
    }

    pub fn teardown(&mut self) -> Result<()> {
        disable_raw_mode()?;
        execute!(self.terminal.backend_mut(), LeaveAlternateScreen)?;
        self.terminal.show_cursor()?;
        Ok(())
    }

    pub fn render_frame(&mut self, processes: &Vec<ProcessInfo>) -> Result<()> {
        self.terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(70), Constraint::Length(0)].as_ref())
                .split(frame.size());
            let rows: Vec<_> = processes
                .iter()
                .map(|p| {
                    let color = get_process_color(p.pid);
                    Row::new([
                        p.pid.to_string(),
                        p.name.clone(),
                        format!("{:.2}%", p.cpu_usage),
                    ])
                    .style(Style::default().fg(color))
                })
                .collect();
            let table = Table::new(rows)
                .block(
                    Block::default()
                        .title("TOPDIO (q: quit)")
                        .borders(Borders::ALL),
                )
                .header(Row::new(["PID", "COMMAND", "CPU USAGE"]))
                .widths(&[
                    Constraint::Length(10),
                    Constraint::Length(45),
                    Constraint::Length(15),
                ])
                .column_spacing(1);
            frame.render_widget(table, chunks[0])
        })?;
        Ok(())
    }
}

impl TopdioSubscriber for UI {
    fn handle(&mut self, message: &TopdioMessage) -> Result<()> {
        match message {
            TopdioMessage::Stats { processes } => self.render_frame(processes),
            TopdioMessage::Stop => self.teardown(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_process_color() {
        assert_eq!(get_process_color(0), Color::LightBlue);
        assert_eq!(get_process_color(5), Color::LightYellow);
        assert_eq!(get_process_color(14), Color::LightGreen);
    }
}
