use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::process::Command;

pub const DESIGN_PROMPT: &str = include_str!("prompt.txt");

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("YSM worker I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("YSM worker exceeded 20 seconds")]
    Timeout,
    #[error("YSM engine did not produce a validated ZIP")]
    InvalidDesign,
}

async fn invoke(reply: &str) -> Result<Vec<u8>, Error> {
    let root = std::env::var_os("YSM_ENGINE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./ysm-engine"));
    let mut child = Command::new(root.join(".venv/bin/python"))
        .arg(root.join("generate.py"))
        .env_clear()
        .env("PYTHONUTF8", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .env("OMP_NUM_THREADS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(reply.as_bytes()).await?;
    }
    let output = child.wait_with_output().await?;
    if !output.status.success() || !output.stdout.starts_with(b"PK\x03\x04") {
        return Err(Error::InvalidDesign);
    }
    Ok(output.stdout)
}

/// Compile a model's structured design using the configured YSM AI Engine worker.
pub async fn package_design(reply: &str) -> Result<Vec<u8>, Error> {
    tokio::time::timeout(Duration::from_secs(20), invoke(reply))
        .await
        .map_err(|_| Error::Timeout)?
}
