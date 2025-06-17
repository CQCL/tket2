//! Hit: CLI tool for exploring rewrite spaces.

mod commands;
mod config;
mod constants;
mod display;
mod factory;
mod storage;

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::fs;
use std::path::PathBuf;

use commands::{
    CheckoutCommand, ChildrenCommand, Command, CommandExecutor, ExtractBestCommand, LoadCommand,
    LogCommand, ParentsCommand, RunCommand, ShowCommand,
};
use constants::HITFILES_DIR;

#[derive(Parser)]
#[command(name = "hit")]
#[command(about = "A CLI tool for constructing and exploring RewriteSpaces using commit factories")]
struct Cli {
    #[command(subcommand)]
    command: CliCommands,
}

#[derive(Subcommand)]
enum CliCommands {
    /// Load a HUGR envelope file and run explorer
    Run {
        /// Input HUGR envelope file
        input_file: PathBuf,
        /// Explorer type
        #[arg(short, long, default_value = "CommuteCZ")]
        factory: String,
        /// Maximum number of rewrites to perform on input
        #[arg(long)]
        max_rewrites: Option<usize>,
    },
    /// Load a rewrite space from timestamp file
    Load {
        /// Timestamp to load (latest if not specified)
        timestamp: Option<String>,
    },
    /// Select commits for extraction
    Checkout {
        /// Commit IDs (can be abbreviated)
        commits: Vec<String>,
    },
    /// Show selected commits and their ancestors
    Log {
        /// Show all commits instead of just selected ones and ancestors
        #[arg(long)]
        all: bool,
    },
    /// Show the current HUGR as a mermaid diagram
    Show,
    /// Extract the best rewrite sequence and select those commits
    ExtractBest,
    /// Show parent commits of a commit
    Parents {
        /// Commit ID (can be abbreviated)
        commit_id: String,
    },
    /// Show child commits of a commit
    Children {
        /// Commit ID (can be abbreviated)
        commit_id: String,
    },
}

impl From<CliCommands> for Command {
    fn from(cmd: CliCommands) -> Self {
        match cmd {
            CliCommands::Run {
                input_file: input_json,
                factory,
                max_rewrites,
            } => Command::Run(RunCommand {
                input_file: input_json,
                factory,
                max_rewrites,
            }),
            CliCommands::Load { timestamp } => Command::Load(LoadCommand { timestamp }),
            CliCommands::Checkout { commits } => Command::Checkout(CheckoutCommand { commits }),
            CliCommands::Log { all } => Command::Log(LogCommand { all }),
            CliCommands::Show => Command::Show(ShowCommand),
            CliCommands::ExtractBest => Command::ExtractBest(ExtractBestCommand),
            CliCommands::Parents { commit_id } => Command::Parents(ParentsCommand { commit_id }),
            CliCommands::Children { commit_id } => Command::Children(ChildrenCommand { commit_id }),
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    // Ensure hitfiles directory exists
    fs::create_dir_all(HITFILES_DIR)?;

    // Convert CLI command to our command enum and execute
    let action: Command = cli.command.into();
    action.execute()
}
