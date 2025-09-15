//! Show command implementation.

use anyhow::{Context, Result};
use hugr::HugrView;
use tket::serialize::TKETDecode;
use tket::Circuit;
use tket_json_rs::SerialCircuit;

use super::CommandExecutor;
use crate::config::Config;
use crate::storage::LoadedRewriteSpace;

#[derive(Debug, Default, Clone, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ShowFormat {
    #[default]
    Mermaid,
    TketJson,
}

#[derive(Debug)]
pub struct ShowCommand {
    pub format: ShowFormat,
}

impl CommandExecutor for ShowCommand {
    fn execute(&self) -> Result<()> {
        // Load the rewrite space data
        let config = Config::load_or_default()?;
        let data = LoadedRewriteSpace::load_from_config(&config)?;

        // Get the current HUGR from selected commits
        let current_hugr = data.current_hugr(&config)?;

        match self.format {
            ShowFormat::Mermaid => {
                // Print the mermaid string
                println!("{}", current_hugr.mermaid_string());
            }
            ShowFormat::TketJson => {
                // Print as TKET1 JSON
                let circ = Circuit::new(current_hugr.to_hugr());
                let ser_circ: SerialCircuit = TKETDecode::encode(&circ)
                    .with_context(|| "could not express HUGR as TKET1 circuit")?;
                serde_json::to_writer(std::io::stdout(), &ser_circ)?;
            }
        }

        Ok(())
    }
}
