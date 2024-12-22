use anyhow::Result;
use crossterm::{
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::io::{stdout, Stdout};
use tui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Modifier, Style},
    text::Text,
    widgets::{Block, Borders, Row, Table},
    Terminal,
};

use crate::topdio::{ProcessInfo, TopdioMessage, TopdioSubscriber};

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

    pub fn render_frame(&mut self, processes: &[ProcessInfo]) -> Result<()> {
        self.terminal.draw(|frame| {
            let chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Length(70), Constraint::Length(0)].as_ref())
                .split(frame.size());
            let rows: Vec<_> = processes
                .iter()
                .map(|p| {
                    Row::new([
                        p.pid.to_string(),
                        p.name.to_str().unwrap_or("(none)").to_string(),
                        format!("{:.2}%", p.cpu_usage),
                    ])
                })
                .collect();
            let table = Table::new(rows)
                .block(Block::default().title("TOPDIO").borders(Borders::ALL))
                .header(Row::new(["PID", "COMMAND", "CPU USAGE"].iter().map(|s| {
                    Text::styled(*s, Style::default().add_modifier(Modifier::BOLD))
                })))
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
