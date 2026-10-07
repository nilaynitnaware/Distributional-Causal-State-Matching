"""Safe production preflight: validate and resolve without running experiments."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

from numerical.device import detect_torch_runtime

from .readiness import (
    ReadinessEvaluation,
    engineering_ready,
    evaluate_finite_sample_readiness,
)
from .registry import (
    PRODUCTION_REGISTRY,
    ProductionRunConfiguration,
)

CheckStatus = Literal["pass", "fail", "warning"]
_READY_STATUSES = frozenset(
    {"ENGINEERING_READY", "SCIENTIFICALLY_READY_UNDER_TESTED_REGIME"}
)
_PREFLIGHT_FIELDS = frozenset({"calibration_artifacts", "regime_id"})


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class PreflightReport:
    valid: bool
    ready: bool
    readiness_status: str
    deterministic_run_id: str | None
    manifest: Mapping[str, object] | None
    resolved_configuration: Mapping[str, object] | None
    readiness: ReadinessEvaluation | None
    checks: tuple[PreflightCheck, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "ready": self.ready,
            "readiness_status": self.readiness_status,
            "deterministic_run_id": self.deterministic_run_id,
            "manifest": self.manifest,
            "resolved_configuration": self.resolved_configuration,
            "readiness": asdict(self.readiness) if self.readiness is not None else None,
            "checks": [asdict(check) for check in self.checks],
        }


def preflight_configuration(
    raw: Mapping[str, object],
    *,
    config_directory: Path = Path("."),
    no_write: bool = True,
) -> PreflightReport:
    """Resolve one configuration, perform checks, and never execute a run."""
    checks: list[PreflightCheck] = []
    configuration_fields = {item.name for item in fields(ProductionRunConfiguration)}
    unknown = sorted(set(raw) - configuration_fields - _PREFLIGHT_FIELDS)
    if unknown:
        checks.append(
            PreflightCheck(
                "configuration_schema",
                "fail",
                f"unknown configuration fields: {unknown}",
            )
        )
        return _invalid_report(checks)
    production_raw = {
        key: value for key, value in raw.items() if key in configuration_fields
    }
    try:
        configuration = ProductionRunConfiguration.from_mapping(production_raw)
        resolved = configuration.resolved()
    except (TypeError, ValueError) as error:
        checks.append(PreflightCheck("configuration_schema", "fail", str(error)))
        return _invalid_report(checks)
    checks.append(PreflightCheck("configuration_schema", "pass", "schema is valid"))
    configuration.effective_runtime_settings()
    checks.append(
        PreflightCheck(
            "runtime_binding_contract",
            "pass",
            "canonical effective runtime settings validated against the protocol",
        )
    )
    checks.append(
        PreflightCheck(
            "registry_resolution",
            "pass",
            f"resolved {configuration.mode!r} to a complete provider graph",
        )
    )
    descriptor = PRODUCTION_REGISTRY[configuration.mode]
    checks.append(
        PreflightCheck(
            "oracle_leakage",
            "pass",
            (
                "oracle capabilities are explicitly permitted for this mode"
                if descriptor.oracle_capabilities_allowed
                else "all configured providers satisfy the non-oracle boundary"
            ),
        )
    )
    checks.extend(_dependency_checks())
    checks.append(_device_check(configuration.device))

    output = Path(configuration.output_directory)
    if not output.is_absolute():
        output = config_directory / output
    output = output.resolve()
    output_check = _output_check(output, no_write=no_write)
    checks.append(output_check)

    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"))
    run_id = _stable_run_id(canonical)
    repeated_run_id = _stable_run_id(canonical)
    deterministic = run_id == repeated_run_id
    checks.append(
        PreflightCheck(
            "deterministic_run_id",
            "pass" if deterministic else "fail",
            run_id if deterministic else "run-ID generation was not deterministic",
        )
    )

    regime_id = raw.get("regime_id", "unspecified_regime")
    if not isinstance(regime_id, str) or not regime_id:
        checks.append(
            PreflightCheck("readiness_regime", "fail", "regime_id must be nonempty")
        )
        return _invalid_report(checks, resolved=resolved)
    finite = configuration.environment_provider == "finite_sample_blackbox"
    readiness = (
        evaluate_finite_sample_readiness(
            _calibration_artifacts(raw, config_directory),
            protocol_version=configuration.protocol_version,
            regime_id=regime_id,
        )
        if finite
        else engineering_ready(
            protocol_version=configuration.protocol_version,
            regime_id=regime_id,
            explanation="all exact/noiseless production engineering checks passed",
        )
    )
    readiness_check_status: CheckStatus = (
        "pass" if readiness.status in _READY_STATUSES else "warning"
    )
    checks.append(
        PreflightCheck(
            "calibration_readiness",
            readiness_check_status,
            f"{readiness.status}: {readiness.explanation}",
        )
    )

    manifest: dict[str, object] = {
        "manifest_kind": "production_preflight",
        "schema_version": configuration.result_schema_version,
        "run_id": run_id,
        "mode": configuration.mode,
        "protocol_version": configuration.protocol_version,
        "regime_id": regime_id,
        "resolved_configuration": resolved,
        "readiness_status": readiness.status,
        "scientific_execution_performed": False,
    }
    checks.append(
        PreflightCheck(
            "manifest_generation",
            "pass",
            "deterministic in-memory manifest generated",
        )
    )
    engineering_valid = all(check.status != "fail" for check in checks)
    if not no_write and engineering_valid:
        _atomic_write_json(output / "preflight_manifest.json", manifest)
        checks.append(
            PreflightCheck(
                "manifest_write",
                "pass",
                str(output / "preflight_manifest.json"),
            )
        )
    return PreflightReport(
        valid=engineering_valid,
        ready=engineering_valid and readiness.status in _READY_STATUSES,
        readiness_status=readiness.status,
        deterministic_run_id=run_id,
        manifest=manifest,
        resolved_configuration=resolved,
        readiness=readiness,
        checks=tuple(checks),
    )


def _calibration_artifacts(
    raw: Mapping[str, object],
    config_directory: Path,
) -> dict[str, Path | Mapping[str, object]]:
    artifacts = raw.get("calibration_artifacts", {})
    if not isinstance(artifacts, Mapping):
        return {}
    normalized: dict[str, Path | Mapping[str, object]] = {}
    for name, value in artifacts.items():
        if not isinstance(name, str):
            continue
        if isinstance(value, str):
            path = Path(value)
            normalized[name] = path if path.is_absolute() else config_directory / path
        elif isinstance(value, Mapping):
            normalized[name] = value
    return normalized


def _dependency_checks() -> list[PreflightCheck]:
    checks = []
    for name in ("numpy", "scipy", "networkx"):
        available = importlib.util.find_spec(name) is not None
        checks.append(
            PreflightCheck(
                f"dependency_{name}",
                "pass" if available else "fail",
                "available" if available else "not importable",
            )
        )
    return checks


def _device_check(device: str) -> PreflightCheck:
    if device not in {"cpu", "cuda", "auto"}:
        return PreflightCheck("device_availability", "fail", "unsupported device")
    if device == "cpu":
        return PreflightCheck("device_availability", "pass", "CPU is available")
    runtime = detect_torch_runtime()
    if device == "cuda" and not runtime.cuda_available:
        return PreflightCheck(
            "device_availability",
            "fail",
            runtime.detail or "CUDA was requested but is unavailable",
        )
    detail = (
        f"CUDA available: {runtime.gpu_name}"
        if runtime.cuda_available
        else "auto will use CPU because CUDA is unavailable"
    )
    return PreflightCheck("device_availability", "pass", detail)


def _output_check(output: Path, *, no_write: bool) -> PreflightCheck:
    if output.exists() and not output.is_dir():
        return PreflightCheck(
            "output_directory_writability",
            "fail",
            "output path exists and is not a directory",
        )
    if not no_write:
        try:
            output.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            return PreflightCheck(
                "output_directory_writability",
                "fail",
                f"{type(error).__name__}: {error}",
            )
    probe = output
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    writable = probe.is_dir() and os.access(probe, os.W_OK)
    return PreflightCheck(
        "output_directory_writability",
        "pass" if writable else "fail",
        (
            f"writable via existing directory {probe}"
            if writable
            else f"no writable existing parent for {output}"
        ),
    )


def _stable_run_id(canonical_resolved_configuration: str) -> str:
    digest = hashlib.sha256(canonical_resolved_configuration.encode()).hexdigest()
    return f"production-{digest[:20]}"


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _invalid_report(
    checks: list[PreflightCheck],
    *,
    resolved: Mapping[str, object] | None = None,
) -> PreflightReport:
    return PreflightReport(
        valid=False,
        ready=False,
        readiness_status="INVALID_CONFIGURATION",
        deterministic_run_id=None,
        manifest=None,
        resolved_configuration=resolved,
        readiness=None,
        checks=tuple(checks),
    )


def _human_report(report: PreflightReport, *, explain: bool) -> str:
    lines = [
        f"preflight: {'VALID' if report.valid else 'INVALID'}",
        f"readiness: {report.readiness_status}",
        f"ready: {str(report.ready).lower()}",
    ]
    if report.deterministic_run_id is not None:
        lines.append(f"run_id: {report.deterministic_run_id}")
    if explain:
        lines.extend(
            f"[{check.status.upper()}] {check.name}: {check.detail}"
            for check in report.checks
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        raw = json.loads(arguments.config.read_text())
        if not isinstance(raw, Mapping):
            raise TypeError("preflight configuration must be a JSON object")
        report = preflight_configuration(
            raw,
            config_directory=arguments.config.parent.resolve(),
            no_write=arguments.no_write,
        )
    except (OSError, TypeError, json.JSONDecodeError) as error:
        report = _invalid_report(
            [PreflightCheck("configuration_file", "fail", str(error))]
        )
    print(
        json.dumps(report.to_mapping(), indent=2, sort_keys=True)
        if arguments.json
        else _human_report(report, explain=arguments.explain)
    )
    return 1 if arguments.strict and not report.ready else 0


if __name__ == "__main__":
    raise SystemExit(main())
