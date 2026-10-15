# KEYSTONE Decision — build guide (Windows + VS Code)

Do the steps in order. Each step ends with a command and the output you should see. If the output differs, stop there and send me what you got.

Every file listed here was run against your `keystone-core.zip` before being handed over: `python -m tests.contracts_test --only D` gives 8 of 8.

---

## Part A — setup

### A1. Unzip and open

1. Right-click `keystone-core.zip` → **Extract All…** → extract to `C:\keystone` (a short path with no spaces).
2. You should now have `C:\keystone\keystone-core\` containing `contracts\`, `tests\`, `README.md`, `requirements-decision.txt`. If you see `C:\keystone\keystone-core\keystone-core\`, use the inner one.
3. VS Code → **File → Open Folder…** → choose `C:\keystone\keystone-core`.
4. **Terminal → New Terminal** (PowerShell). The prompt should end in `keystone-core>`.

Check:

    dir

You should see `contracts`, `tests`, `requirements-decision.txt` and the `.md` files. There is no `decision` folder yet.

### A2. Find your Python

    py -0

Lists installed versions, e.g. ` -V:3.11 *  Python 3.11 (64-bit)`. Any of 3.10, 3.11, 3.12 is fine. If `py` is not recognised, install Python 3.11 from python.org (tick "Add python.exe to PATH") and reopen VS Code.

### A3. Virtual environment

    py -3.11 -m venv .venv
    .venv\Scripts\activate

(use your version number if it is not 3.11). The prompt now starts with `(.venv)`.

If activation is blocked with "running scripts is disabled":

    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    .venv\Scripts\activate

Then in VS Code: `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one containing `.venv`.

### A4. Install

    pip install -r requirements-decision.txt

Takes a few minutes. Last line starts `Successfully installed …` and includes `numpy`, `pandas`, `plotly`, `streamlit`.

### A5. See the problem

    python -m contracts.fakes

Look at the `DECISION SERIES` table: `centre_upper` climbs to 9.5 mm on 2026-07-14 and is back near zero by October, while `east_upper` reads 8.0, 11.3, 13.5, 17.7 on the last four rows and never returns.

### A6. Baseline tests

    python -m tests.contracts_test

Last lines:

    3 passed, 0 failed, 3 pending
    everything implemented so far conforms

Setup is done.

---

## Part B — build, one file at a time

Create the folder first: in the VS Code Explorer, right-click the empty area → **New Folder** → `decision`. It sits next to `contracts` and `tests`.

For each file: right-click `decision` → **New File** → type the name → paste the whole contents of the file I sent → `Ctrl+S`. (Or extract `keystone-decision.zip` into `C:\keystone\keystone-core` to get all of them at once, and just run the checks.)

Always run commands from `C:\keystone\keystone-core` with `(.venv)` active.

### B1. `decision/__init__.py`

    python -c "import decision; print(decision.__version__)"

Output: `1.0.0`

### B2. `decision/seasonal.py`

The seasonal model: expected movement = a × Δtemperature + b × Δrainfall, fitted on the other regions at the same height, never on the region being judged.

    python -m decision.seasonal

You should see the fitted coefficient next to the true one for each region (upper regions about 0.7–0.8 mm/°C, lower about 0.35), each region's peers listed **without itself**, and then:

    centre_upper: observed vs expected (band = 2 sigma)
      2026-03-01   29.7 C   observed    2.0   expected    2.7 +/- 3.2
      ...
      2026-07-14   33.4 C   observed    9.5   expected    5.6 +/- 3.8

The summer peak is expected, not alarming.

### B3. `decision/residuals.py`

    python -m decision.residuals

`centre_upper` residuals wander between about −3.7 and +3.6 with `persistent: False`. `east_upper` ends:

      2026-10-12        8.0      0.6      7.4    5.4
      2026-11-26       11.3      1.9      9.4    5.8
      2027-01-10       13.5     -0.2     13.7   10.2
      2027-02-24       17.7      2.2     15.5   10.9
      run of strong residuals: 4   retained: True   trend: +2.9 mm/capture   persistent: True

### B4. `decision/severity.py`

    python -m decision.severity

A table of severities after each capture. The `east_upper` column reads INFORMATIONAL for reports 0–4, then `WATCH`, `ALERT`, `CRITICAL`, `CRITICAL`. No other column ever goes above `WATCH`.

### B5. `decision/routing.py`

    python -m decision.routing

Four routes, each with a different recipient:

    INFORMATIONAL  -> Monument record
    WATCH          -> Capture schedule
    ALERT          -> Regional office priority list   [notifies a person]
    CRITICAL       -> Structural engineer + site closure recommendation   [notifies a person]

### B6. `decision/store.py`

    python -m decision.store

    ARCHIVE self-test  (C:\Users\...\keystone_selftest.db)
      stored   9 reports, 54 alerts
      re-saved everything: no duplicates
      reports read back identical; re-evaluating from the archive gives identical alerts
      monuments: ['M-01']
      ALERT or higher, in order:
        2026-11-26  east_upper    ALERT     -> Regional office priority list
        2027-01-10  east_upper    CRITICAL  -> Structural engineer + site closure recommendation
        2027-02-24  east_upper    CRITICAL  -> Structural engineer + site closure recommendation
      archive closed and test file removed

(`store.py` imports `pipeline.py` for this self-test, so if you are pasting file by file, create B7 first and then run this check.)

### B7. `decision/pipeline.py`

    python -m decision.pipeline

The same severity table as B4, then six final alerts. The last one begins:

    [CRITICAL] east_upper -> Structural engineer + site closure recommendation
      15.5 mm further outward than the seasonal model expects (expected +2.2 mm, measured +17.7 mm, ...

and its text says KEYSTONE *recommends* that the custodian consider restricting access and has no authority to close a monument.

**Now the acceptance tests:**

    python -m tests.contracts_test --only D

    WORKSTREAM D — decision
      PASS     one valid Alert per region, JSON-safe
      PASS     alerts name the captures that caused them
      PASS     no ALERT with fewer than three captures
      PASS     summer heat never reaches ALERT (10 seeds, every prefix)
      PASS     structural drift caught, stable regions stay calm
      PASS     no ALERT before the drift begins
      PASS     drift in a load-bearing region escalates
      PASS     crack widening from the crack-seg team is acted on

    8 passed, 0 failed, 0 pending

Then the whole suite: `python -m tests.contracts_test` → `11 passed, 0 failed, 2 pending` (the 2 pending are Geometry's).

### B8. `decision/dashboard.py`

    streamlit run decision/dashboard.py

First run asks for an email — press Enter to skip. A browser tab opens at `http://localhost:8501`. You should see four counters (5 INFORMATIONAL, 0 WATCH, 0 ALERT, 1 CRITICAL), and six charts: `east_upper`'s blue line leaves the grey seasonal band from October and keeps climbing, while `centre_upper` crosses the dotted red naive threshold in July but stays inside the band.

Things to try: set **Region with structural drift** to "none" (everything turns INFORMATIONAL even in summer); drag **Replay: captures so far** left to watch WATCH → ALERT → CRITICAL build up; press **Save this history to the archive**, then switch the data source to **Archive (SQLite)**.

Stop the server with `Ctrl+C` in the terminal.

### B9. `decision/benchmark.py` (extra — produces the write-up's numbers)

    python -m decision.benchmark --seeds 50      (about half a minute)
    python -m decision.benchmark                 (200 years per experiment, a few minutes)

The first block of the full run ends with:

    (first 20 years only, naive threshold: 10 false alarms in 5 of 20 years)

and the KEYSTONE row shows `0 / 200`. `WRITEUP.md` is built from this output.

---

## If something goes wrong

- `ModuleNotFoundError: No module named 'contracts'` — you are not in `C:\keystone\keystone-core`. Run `cd C:\keystone\keystone-core`.
- `ModuleNotFoundError: No module named 'numpy'` (or streamlit) — the venv is not active. Run `.venv\Scripts\activate`.
- `ImportError` naming a `decision.` module — that file is missing or misnamed; check the `decision` folder has exactly the file names above.
- `streamlit` is not recognised — use `python -m streamlit run decision/dashboard.py`.
- Add `keystone.db` to `.gitignore` if you commit; do not edit anything in `contracts/`.
