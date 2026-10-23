use cubecl_environment::records::Stamped;
use cubecl_server::compiler::CompilationRecord;
use std::{error::Error, fs::File};

fn main() -> Result<(), Box<dyn Error>> {
    for path in std::env::args().skip(1) {
        let record: Stamped<CompilationRecord> = ciborium::de::from_reader(File::open(&path)?)?;
        println!(
            "{}",
            serde_json::json!({
                "path": path,
                "record": {
                    "stamp": record.stamp,
                    "record": {
                        "kernel": record.record.kernel,
                        "key_debug": format!("{:?}", record.record.key),
                        "outcome": format!("{:?}", record.record.outcome),
                        "duration": record.record.duration,
                        "ir": record.record.ir,
                        "source": record.record.source,
                    },
                },
            })
        );
    }
    Ok(())
}
