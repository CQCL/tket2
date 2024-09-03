use std::error::Error;
use std::io::Write;

use clap::{Parser, ValueEnum};
use clio::Output;
use serde_json::to_string_pretty;

use hugr::{algorithms::validation::ValidationLevel, extension::ExtensionRegistry, hugr::hugrmut::HugrMut, Hugr, HugrView};
use hugr_cli::{HugrArgs, Package};

use crate::{lazify_measure::LazifyMeasurePass, HSeriesPass};

/// CLI arguments.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "HUGR CLI tools.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub enum CliArgs {
    Run(RunArgs),
}

#[derive(Copy,Clone, PartialEq,Eq,PartialOrd,Ord,Debug,ValueEnum)]
pub enum Pass {
    All,
    LazifyMeasure,
    ForceOrder,
}



/// Validate and visualise a HUGR file.
#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct RunArgs {
    #[command(flatten)]
    /// common arguments
    pub hugr_args: HugrArgs,

    #[arg(value_enum, default_value = "all")]
    pub pass: Pass,

    #[arg(long="validate",default_value = "true")]
    pub validate_pass: bool,
}

#[derive(Parser, Debug)]
#[clap(version = "1.0", long_about = None)]
#[clap(about = "Validate a HUGR.")]
#[group(id = "hugr")]
#[non_exhaustive]
pub struct RunCommand {
    #[command(flatten)]
    pub run_args: RunArgs,

    /// Output file '-' for stdout
    #[clap(long, short, value_parser, default_value = "-")]
    output: Output,
}

impl RunArgs {
    pub fn run(&mut self) -> Result<Hugr, Box<dyn Error>>{
        let (hugrs, reg) = self.hugr_args.validate()?;
        let Ok::<[_;1],_>([mut hugr]) = hugrs.try_into() else {
            Err("Only one module is supported".to_string())?
        };
        let validation_level = if self.validate_pass {
            ValidationLevel::WithExtensions
        } else {
            ValidationLevel::WithoutExtensions
        };

        self.pass.run(&mut hugr, &reg, validation_level)?;
        Ok(hugr)
    }
}

impl RunCommand {
    pub fn run(&mut self) -> Result<(), Box<dyn Error>> {
        write!(self.output, "{}", serde_json::to_string_pretty(&self.run_args.run()?)?)?;
        Ok(())
    }
}

impl Pass {
    pub fn run(&self, hugr: &mut impl HugrMut, registry: &ExtensionRegistry, level: ValidationLevel) -> Result<(), Box<dyn Error>> {
        let pass = HSeriesPass::default().with_validation_level(level);
        match self {
            Pass::All => pass.run(hugr, registry)?,
            Pass::LazifyMeasure => pass.lazify_measure(hugr, registry)?,
            Pass::ForceOrder => pass.force_order(hugr, registry)?,
        };
        Ok(())
    }
}
