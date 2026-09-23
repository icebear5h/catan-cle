from scripts.verify_data_pipeline_layout import DEFAULT_MANIFEST, verify_layout


def test_data_pipeline_artifact_layout_receipt() -> None:
    summary = verify_layout(DEFAULT_MANIFEST)

    assert summary == {
        "groups": 15,
        "files": 352,
        "bytes": 66_402_952,
        "duplicates_removed": 5,
    }
