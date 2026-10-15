"""KEYSTONE Workstream D - Decision.

Reads the history of ChangeReports for one monument and decides, for every
region, whether its movement is ordinary seasonal breathing or persistent
structural movement - and who should be told.

    seasonal.py    what each region SHOULD have moved, given the weather,
                   learned from its neighbours at the same height
    residuals.py   actual minus expected, and whether that gap persists
    severity.py    INFORMATIONAL / WATCH / ALERT / CRITICAL
    routing.py     severity -> recipient and recommended action
    store.py       SQLite archive of every report and alert
    pipeline.py    evaluate(history) -> list[Alert]   (the entry point)
    dashboard.py   streamlit run decision/dashboard.py
    benchmark.py   python -m decision.benchmark  (numbers for the write-up)

Nothing in this package edits or depends on anything outside contracts/.
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
