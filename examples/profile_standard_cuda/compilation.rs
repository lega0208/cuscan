use cubecl::records::{MarkRecord, Session, Stamped};
use cubecl_environment::persistence::Database;
use cubecl_server::compiler::{CompilationOutcome, CompilationRecord};
use std::sync::Mutex;

struct WarningCapture;
static WARNINGS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOGGER: WarningCapture = WarningCapture;

impl log::Log for WarningCapture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let message = format!("{}: {}", record.target(), record.args());
            if message.contains("cubecl_environment")
                || message.contains("Unable to open the Turso cache")
            {
                WARNINGS.lock().unwrap().push(message.clone());
            }
            eprintln!("{message}");
        }
    }

    fn flush(&self) {}
}

pub fn install_warning_capture() -> Result<(), Box<dyn std::error::Error>> {
    log::set_logger(&LOGGER).map_err(|error| {
        std::io::Error::other(format!("cannot install CubeCL warning capture: {error}"))
    })?;
    log::set_max_level(log::LevelFilter::Warn);
    Ok(())
}

fn decoded<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, String> {
    ciborium::from_reader(bytes).map_err(|error| error.to_string())
}

#[derive(Debug)]
struct CompilationSample {
    offset: std::time::Duration,
    duration: std::time::Duration,
    outcome: CompilationOutcome,
}

fn phase_intervals(
    labels: &[String],
    marks: &[(String, std::time::Duration, std::time::Duration)],
) -> Result<std::collections::BTreeMap<String, (std::time::Duration, std::time::Duration)>, String>
{
    let mut intervals = std::collections::BTreeMap::new();
    for label in labels {
        let matching = marks
            .iter()
            .filter(|(mark_label, _, _)| mark_label == label)
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!(
                "phase {label} has {} retained marks",
                matching.len()
            ));
        }
        let (_, start, wall) = matching[0];
        if wall.is_zero() {
            return Err(format!("phase {label} has a zero or invalid mark duration"));
        }
        let end = start
            .checked_add(*wall)
            .ok_or_else(|| format!("phase {label} interval overflows"))?;
        intervals.insert(label.clone(), (*start, end));
    }
    Ok(intervals)
}

fn summarize_compilations(
    labels: &[String],
    intervals: &std::collections::BTreeMap<String, (std::time::Duration, std::time::Duration)>,
    samples: &[CompilationSample],
) -> Result<
    (
        serde_json::Map<String, serde_json::Value>,
        [f64; 3],
        [usize; 3],
        usize,
    ),
    String,
> {
    let mut phase_json = serde_json::Map::new();
    let mut outcome_seconds = [0.0f64; 3];
    let mut outcome_counts = [0usize; 3];
    let mut covered_records = 0usize;
    for phase in labels {
        let (start, end) = intervals
            .get(phase)
            .copied()
            .ok_or_else(|| format!("phase {phase} has no retained mark"))?;
        let records = samples
            .iter()
            .filter(|sample| sample.offset >= start && sample.offset < end)
            .collect::<Vec<_>>();
        let mut seconds = [0.0f64; 3];
        let mut counts = [0usize; 3];
        for sample in &records {
            let Some(record_end) = sample.offset.checked_add(sample.duration) else {
                return Err(format!(
                    "phase {phase} contains an overflowing compilation record"
                ));
            };
            if !sample.duration.as_secs_f64().is_finite() || record_end > end {
                return Err(format!(
                    "phase {phase} contains an invalid or out-of-bound compilation record"
                ));
            }
            let index = match sample.outcome {
                CompilationOutcome::Compiled => 0,
                CompilationOutcome::Loaded => 1,
                CompilationOutcome::Rekeyed => 2,
            };
            let duration = sample.duration.as_secs_f64();
            seconds[index] += duration;
            counts[index] += 1;
            outcome_seconds[index] += duration;
            outcome_counts[index] += 1;
            covered_records += 1;
        }
        phase_json.insert(phase.clone(), serde_json::json!({
            "start_seconds_from_session": start.as_secs_f64(),
            "end_seconds_from_session": end.as_secs_f64(),
            "record_count": records.len(),
            "outcome_record_counts": {"compiled": counts[0], "loaded": counts[1], "rekeyed": counts[2]},
            "outcome_seconds": {"compiled": seconds[0], "loaded": seconds[1], "rekeyed": seconds[2]}
        }));
    }
    if covered_records != samples.len() {
        return Err(format!(
            "{} retained compilation records fall outside requested phase intervals",
            samples.len() - covered_records
        ));
    }
    Ok((phase_json, outcome_seconds, outcome_counts, covered_records))
}

pub fn capture(
    environment_name: &str,
    label: &str,
    environment_root: &std::path::Path,
    phase_labels: &[String],
    warmups: usize,
) -> serde_json::Value {
    let warnings = WARNINGS.lock().unwrap().clone();
    let mut report = serde_json::json!({
        "status": "unavailable",
        "scope": "artifact_acquisition_excluding_kernel_execution",
        "duration_definition": "CompilationRecord.duration measures kernel artifact acquisition through device load, including cache/codegen/compiler/NVRTC/persistence costs where applicable; kernel execution is excluded.",
        "compiled_seconds": null,
        "loaded_seconds": null,
        "rekeyed_seconds": null,
        "phase_records": [],
        "requested_phases": phase_labels,
        "records_config": {"level": "basic", "keep_sessions": null},
        "coverage": {"status": "unavailable", "reason": "capture did not complete"},
        "environment": {
            "name": environment_name,
            "root": environment_root,
            "identity": "fresh_run_specific_environment"
        },
        "native_evidence": {"fresh_environment_selected": true, "database_opened": false},
        "persistence_warnings": warnings
    });

    let result = (|| -> Result<serde_json::Value, String> {
        if !warnings.is_empty() {
            return Err("CubeCL persistence warning observed during profiling".into());
        }
        let database =
            Database::open_active().ok_or("active CubeCL database could not be opened")?;
        let summary = database.summary();
        let summary_counts = summary
            .iter()
            .map(|entry| (entry.namespace.clone(), entry.entries))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut raw_counts = std::collections::BTreeMap::<String, usize>::new();
        let mut sessions = Vec::<Session>::new();
        let mut compilations = Vec::<Stamped<CompilationRecord>>::new();
        let mut marks = Vec::<Stamped<MarkRecord>>::new();
        for (namespace, kind) in [
            ("records/v1/sessions", "sessions"),
            ("records/v1/compilation", "compilation"),
            ("records/v1/marks", "marks"),
        ] {
            let mut values = Vec::<Vec<u8>>::new();
            database.scan(namespace, &mut |_, value| values.push(value.to_vec()));
            raw_counts.insert(namespace.to_string(), values.len());
            for value in values {
                match kind {
                    "sessions" => sessions.push(decoded(&value)?),
                    "compilation" => compilations.push(decoded(&value)?),
                    "marks" => marks.push(decoded(&value)?),
                    _ => unreachable!(),
                }
            }
        }
        for namespace in [
            "records/v1/sessions",
            "records/v1/compilation",
            "records/v1/marks",
        ] {
            let expected = summary_counts.get(namespace).copied().unwrap_or(0);
            let observed = raw_counts.get(namespace).copied().unwrap_or(0) as u64;
            if expected != observed {
                return Err(format!(
                    "database summary/scan mismatch for {namespace}: {expected} vs {observed}"
                ));
            }
        }
        let matching = sessions
            .iter()
            .filter(|session| {
                session.label.as_deref() == Some(label) && session.process == std::process::id()
            })
            .collect::<Vec<_>>();
        if matching.len() != 1 {
            return Err(format!(
                "expected one retained matching session, found {}",
                matching.len()
            ));
        }
        let session_id = matching[0].id;
        let session_compilations = compilations
            .into_iter()
            .filter(|record| record.stamp.session == session_id)
            .collect::<Vec<_>>();
        let session_marks = marks
            .into_iter()
            .filter(|record| record.stamp.session == session_id)
            .collect::<Vec<_>>();
        if session_compilations.is_empty()
            || !session_compilations
                .iter()
                .any(|record| record.record.outcome == CompilationOutcome::Compiled)
        {
            return Err(
                "no retained freshly stored compilation record; observed zero is not established"
                    .into(),
            );
        }
        let marks = session_marks
            .iter()
            .map(|mark| {
                (
                    mark.record.label.clone(),
                    mark.stamp.offset,
                    mark.record.wall,
                )
            })
            .collect::<Vec<_>>();
        let phase_intervals = phase_intervals(phase_labels, &marks)?;
        let samples = session_compilations
            .iter()
            .map(|record| CompilationSample {
                offset: record.stamp.offset,
                duration: record.record.duration,
                outcome: record.record.outcome,
            })
            .collect::<Vec<_>>();
        let (phase_json, outcome_seconds, outcome_counts, covered_records) =
            summarize_compilations(phase_labels, &phase_intervals, &samples)?;
        let measured_compilations = session_compilations
            .iter()
            .filter(|record| {
                phase_intervals.iter().any(|(phase, (start, end))| {
                    phase.starts_with("measured_")
                        && record.stamp.offset >= *start
                        && record.stamp.offset < *end
                })
            })
            .collect::<Vec<_>>();
        let measured_compiled_count = measured_compilations
            .iter()
            .filter(|record| record.record.outcome == CompilationOutcome::Compiled)
            .count();
        let final_warnings = WARNINGS.lock().unwrap().clone();
        if !final_warnings.is_empty() {
            return Err("CubeCL persistence warning observed during record read".into());
        }
        Ok(serde_json::json!({
            "status": "measured",
            "scope": "artifact_acquisition_excluding_kernel_execution",
            "duration_definition": "CompilationRecord.duration measures kernel artifact acquisition through device load, including cache/codegen/compiler/NVRTC/persistence costs where applicable; kernel execution is excluded.",
            "compiled_seconds": outcome_seconds[0],
            "loaded_seconds": outcome_seconds[1],
            "rekeyed_seconds": outcome_seconds[2],
            "outcome_record_counts": {"compiled": outcome_counts[0], "loaded": outcome_counts[1], "rekeyed": outcome_counts[2]},
            "phase_records": phase_json,
            "session_id": session_id,
            "process_id": matching[0].process,
            "session_label": label,
            "environment": {"name": environment_name, "root": environment_root, "identity": "fresh_run_specific_environment"},
            "native_evidence": {"fresh_environment_selected": true, "database_opened": true, "raw_cbor_records_decoded": true},
            "requested_phases": phase_labels,
            "records_config": {"level": "basic", "keep_sessions": null},
            "warmup_compilation_check": {
                "warmups_requested": warmups,
                "measured_phase_compiled_record_count": measured_compiled_count,
                "fresh_compilation_after_warmup": warmups > 0 && measured_compiled_count > 0
            },
            "coverage": {
                "status": "verified_observed_records",
                "retained_session": true,
                "all_raw_values_decoded": true,
                "summary_counts_match_raw_scan": true,
                "phase_marks_complete": true,
                "records_covered_by_requested_phases": covered_records,
                "compilation_records_in_session": session_compilations.len(),
                "database_namespace_entry_counts": summary_counts,
                "raw_namespace_entry_counts": raw_counts,
                "limitations": "CubeCL persistence scan APIs are best-effort and may log failures; explicit warning capture, raw/summary count comparisons, decoding, session identity, and phase coverage reject observed incompleteness but cannot prove lossless storage under every failure mode."
            },
            "persistence_warnings": final_warnings
        }))
    })();
    match result {
        Ok(value) => value,
        Err(reason) => {
            report["coverage"] = serde_json::json!({"status": "incomplete", "reason": reason});
            report["unavailable_reason"] = serde_json::Value::String(reason);
            report["persistence_warnings"] = serde_json::json!(WARNINGS.lock().unwrap().clone());
            report
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seconds(value: u64) -> std::time::Duration {
        std::time::Duration::from_secs(value)
    }

    #[test]
    fn corrupt_record_decode_is_an_error() {
        assert!(decoded::<MarkRecord>(&[0xff, 0x00]).is_err());
    }

    #[test]
    fn missing_phase_marker_is_an_error() {
        let labels = vec!["warmup_0".to_string(), "measured_fit_0".to_string()];
        let marks = vec![("warmup_0".to_string(), seconds(0), seconds(5))];
        assert!(phase_intervals(&labels, &marks).is_err());
    }

    #[test]
    fn incomplete_compilation_coverage_is_an_error() {
        let labels = vec!["warmup_0".to_string()];
        let intervals = [("warmup_0".to_string(), (seconds(0), seconds(5)))]
            .into_iter()
            .collect();
        let samples = vec![CompilationSample {
            offset: seconds(6),
            duration: seconds(1),
            outcome: CompilationOutcome::Compiled,
        }];
        assert!(summarize_compilations(&labels, &intervals, &samples).is_err());
    }

    #[test]
    fn retained_warmup_compile_allows_measured_phase_zero() {
        let labels = vec!["warmup_0".to_string(), "measured_fit_0".to_string()];
        let marks = vec![
            ("warmup_0".to_string(), seconds(0), seconds(5)),
            ("measured_fit_0".to_string(), seconds(5), seconds(5)),
        ];
        let intervals = phase_intervals(&labels, &marks).unwrap();
        let samples = vec![CompilationSample {
            offset: seconds(1),
            duration: seconds(2),
            outcome: CompilationOutcome::Compiled,
        }];
        let (phases, total_seconds, total_counts, covered) =
            summarize_compilations(&labels, &intervals, &samples).unwrap();

        assert_eq!(total_seconds[0], 2.0);
        assert_eq!(total_counts[0], 1);
        assert_eq!(covered, 1);
        assert_eq!(phases["measured_fit_0"]["record_count"], 0);
        assert_eq!(phases["measured_fit_0"]["outcome_seconds"]["compiled"], 0.0);
    }

    #[test]
    fn phase_boundaries_and_outcome_durations_are_accounted_once() {
        let labels = vec!["phase_a".to_string(), "phase_b".to_string()];
        let marks = vec![
            ("phase_a".to_string(), seconds(0), seconds(10)),
            ("phase_b".to_string(), seconds(10), seconds(10)),
        ];
        let intervals = phase_intervals(&labels, &marks).unwrap();
        let samples = vec![
            CompilationSample {
                offset: seconds(0),
                duration: seconds(2),
                outcome: CompilationOutcome::Compiled,
            },
            CompilationSample {
                offset: seconds(8),
                duration: seconds(2),
                outcome: CompilationOutcome::Loaded,
            },
            CompilationSample {
                offset: seconds(10),
                duration: seconds(1),
                outcome: CompilationOutcome::Rekeyed,
            },
        ];
        let (phases, totals, counts, covered) =
            summarize_compilations(&labels, &intervals, &samples).unwrap();

        assert_eq!(covered, 3);
        assert_eq!(counts, [1, 1, 1]);
        assert_eq!(totals, [2.0, 2.0, 1.0]);
        assert_eq!(phases["phase_a"]["record_count"], 2);
        assert_eq!(phases["phase_b"]["record_count"], 1);
        assert_eq!(phases["phase_b"]["outcome_record_counts"]["rekeyed"], 1);
    }
}
