# Integration tests

End-to-end tests that exercise the full pipeline on small fixtures.

Run with:

```bash
pytest tests/integration/
```

These tests serve as smoke tests for releases: if any integration test
fails, the build is not release-ready.
