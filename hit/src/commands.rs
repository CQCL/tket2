//! Command implementations for the hit CLI tool.

use anyhow::Result;
use enum_dispatch::enum_dispatch;

pub mod checkout;
pub mod children;
pub mod extract_best;
pub mod load;
pub mod log;
pub mod parents;
pub mod show;

pub use checkout::CheckoutCommand;
pub use children::ChildrenCommand;
pub use extract_best::ExtractBestCommand;
pub use load::LoadCommand;
pub use log::LogCommand;
pub use parents::ParentsCommand;
pub use show::ShowCommand;

/// Trait for command execution
#[enum_dispatch]
pub trait CommandExecutor {
    /// Execute the command
    fn execute(&self) -> Result<()>;
}

/// Enum representing all possible commands
#[enum_dispatch(CommandExecutor)]
#[derive(Debug)]
pub enum Command {
    Load(LoadCommand),
    Checkout(CheckoutCommand),
    Log(LogCommand),
    Show(ShowCommand),
    ExtractBest(ExtractBestCommand),
    Parents(ParentsCommand),
    Children(ChildrenCommand),
}
