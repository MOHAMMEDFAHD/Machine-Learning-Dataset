# Changelog

## 2026-11-11

### Fixed
- **Raw CSVs regenerated.** The shipped `retail_raw.csv` had been re-saved by a spreadsheet app: 992 ISO dates were rewritten as `M/D/YYYY`, and 14 deliberate day-first dates (for example `05/01/2026`) were silently turned into wrong dates, so the date exercise no longer matched the defect manifest. `healthcare_raw.csv` had only cosmetic changes (`29.0` → `29`, CRLF line endings). Both files now match `inject_defects.py` byte for byte. Clean files and the manifest are unchanged.
- CSV writers now force `\n` line endings, and `.gitattributes` marks CSV and notebook files as binary-safe, so files are identical on Windows, macOS and Linux.
- `model_examples.py` no longer relies on `select_dtypes(include='object')`, which warns under pandas 3.

### Added
- `src/clean_data.py`: dictionary-driven reference cleaning solution with a repair log and a grade against the clean answer key. All intentional defects stay in the raw files; this script shows one correct way to repair them.
- Retail reference scenarios B (same-day plan with opening inventory, price, discount, advertising and promotion) and C (labelled post-outcome leakage demo).
- Tests: raw files must match the injector; every teaching defect must be present; the reference cleaning must produce zero wrong values outside the ID column.
- README: defect table, reference results table and a warning against editing raw CSVs in spreadsheet apps.
