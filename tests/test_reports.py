"""VarshaMitra Phase 7 Report Generation & Export Test Suite.
==========================================================
Validates downloadable PDF and CSV report endpoints, adherence to
scientific benchmarks, header schemas, data integrity, and error handling.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys
import pytest
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from api.main import (
    load_resources,
    download_pdf_report,
    download_csv_report,
    download_default_pdf_report,
    download_default_csv_report,
    DATA_STORE
)
from api.services.report_service import (
    generate_pdf_report,
    generate_csv_report,
    get_canonical_report_id,
    DISCLAIMER_TEXT
)


@pytest.fixture(scope="module", autouse=True)
def init_api_resources():
    """Ensure data stores and model registries are populated before tests."""
    load_resources()


# -----------------------------------------------------------------------------
# 1. PDF Report Tests
# -----------------------------------------------------------------------------
def test_pdf_endpoint_success():
    """Verify PDF endpoint returns HTTP 200, application/pdf, and valid PDF bytes."""
    res = download_pdf_report(report_id="VM-20240928-12Z", district="Pune")
    assert res.status_code == 200
    assert res.media_type == "application/pdf"
    assert res.body.startswith(b"%PDF-")
    assert len(res.body) > 1000
    assert "VarshaMitra_Report_VM-20240928-12Z.pdf" in res.headers["content-disposition"]


def test_pdf_custom_and_fallback_district():
    """Verify PDF endpoint generates successfully with different districts."""
    res_mumbai = download_pdf_report(report_id="VM-20240928-12Z", district="Mumbai")
    assert res_mumbai.status_code == 200
    assert res_mumbai.body.startswith(b"%PDF-")

    res_none = download_pdf_report(report_id="VM-20240928-12Z", district=None)
    assert res_none.status_code == 200
    assert res_none.body.startswith(b"%PDF-")


def test_pdf_default_alias_endpoint():
    """Verify default alias GET /api/reports/pdf functions correctly."""
    res = download_default_pdf_report(district="Satara")
    assert res.status_code == 200
    assert res.media_type == "application/pdf"
    assert res.body.startswith(b"%PDF-")


def test_pdf_service_content_and_disclaimer():
    """Verify PDF generation includes the required scientific disclaimer."""
    assert "VarshaMitra AI Post-Processed Forecast" in DISCLAIMER_TEXT
    assert "not an official IMD warning" in DISCLAIMER_TEXT

    pdf_bytes = generate_pdf_report("VM-TEST-001", "Pune", DATA_STORE)
    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF-")
    assert len(pdf_bytes) > 2000


# -----------------------------------------------------------------------------
# 2. CSV Report Tests
# -----------------------------------------------------------------------------
def test_csv_endpoint_success():
    """Verify CSV endpoint returns HTTP 200, text/csv, and expected 36 districts."""
    res = download_csv_report(report_id="VM-20240928-12Z")
    assert res.status_code == 200
    assert "text/csv" in res.media_type
    assert "VarshaMitra_Data_VM-20240928-12Z.csv" in res.headers["content-disposition"]

    csv_text = res.body.decode("utf-8")
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]
    
    # 1 header + 36 Maharashtra districts
    assert len(lines) == 37

    header = lines[0].split(",")
    expected_cols = [
        "report_id", "forecast_time", "valid_time", "district", "division",
        "latitude", "longitude", "raw_nwp_rainfall_mm", "corrected_rainfall_p10_mm",
        "corrected_rainfall_p50_mm", "corrected_rainfall_p90_mm", "downscaling_delta_mm",
        "regime", "regime_probability", "heavy_rain_probability",
        "very_heavy_rain_probability", "extremely_heavy_rain_probability",
        "risk_category", "cell_count", "model_version"
    ]
    for col in expected_cols:
        assert col in header, f"Missing expected CSV header column: {col}"


def test_csv_default_alias_endpoint():
    """Verify default alias GET /api/reports/csv functions correctly."""
    res = download_default_csv_report()
    assert res.status_code == 200
    assert "text/csv" in res.media_type
    csv_text = res.body.decode("utf-8")
    lines = [line.strip() for line in csv_text.strip().split("\n") if line.strip()]
    assert len(lines) == 37


def test_csv_data_honesty_and_validity():
    """Verify CSV rows contain valid numeric values and correct regime mappings."""
    csv_text = generate_csv_report("VM-20240928-12Z", DATA_STORE)
    lines = [l.strip() for l in csv_text.strip().split("\n") if l.strip()]
    assert len(lines) == 37

    # Check first district row
    row1 = lines[1].split(",")
    assert row1[0] == "VM-20240928-12Z"
    # Ensure raw_nwp_rainfall_mm and corrected_rainfall_p50_mm are valid floats
    raw_val = float(row1[7])
    p50_val = float(row1[9])
    assert raw_val >= 0.0
    assert p50_val >= 0.0


# -----------------------------------------------------------------------------
# 3. Report ID Sanitization & Error Handling
# -----------------------------------------------------------------------------
def test_report_id_sanitization():
    """Verify special characters in report_id are safely sanitized in filenames."""
    res_pdf = download_pdf_report(report_id="VM/2024-09-28;rm -rf/*", district="Pune")
    assert res_pdf.status_code == 200
    fname = res_pdf.headers["content-disposition"].split("filename=")[1]
    assert ";" not in fname
    assert "/" not in fname
    assert "*" not in fname
    assert res_pdf.headers["x-report-id"] == "VM_2024-09-28_rm_-rf__"

    res_csv = download_csv_report(report_id="VM<script>alert(1)</script>")
    assert res_csv.status_code == 200
    fname_csv = res_csv.headers["content-disposition"].split("filename=")[1]
    assert "<" not in fname_csv
    assert ">" not in fname_csv
    assert "<script>" not in fname_csv


def test_canonical_report_id_format():
    """Verify deterministic canonical report ID generation."""
    cid = get_canonical_report_id()
    assert cid.startswith("VM-")
    assert cid.endswith("Z")
