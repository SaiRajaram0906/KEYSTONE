# 5-minute demo script (simulated data; say so early)
Run: `streamlit run console/app.py` from the project root. Keep default settings (numbers below are for the defaults).

0:00 Problem [title + CAG slide]: 3,698 protected monuments, no complete physical survey since Independence (CAG audit), 92 of 1,655 inspected could not be traced. KEYSTONE measures a monument again and again and raises its own alarm.
0:40 Idea [Whole chain tab]: photograph, quality-check, scaled 3D model, compare with last visit, separate seasonal breathing from real damage, act. Six cards = six stages, real code.
1:20 Honest framing [banner]: the wall is simulated with known truth so every number can be checked.
2:00 Change in 3D: planted 12 mm, measured 11.8 mm, scale error 0.01%. Grey points are inside the noise and ignored.
2:50 Cracks: widening of 2 mm read as 2.2 mm (about 1 px). Model trained on concrete; brick walls are weaker (IoU 0.455).
3:20 Temple, one year: tower leans from month 2, front-right column settles from month 4, beam and plinth cracks open. KEYSTONE alerts on the tower at month 7 (fixed threshold: month 10); beam and plinth caught (fixed threshold never). Hairline control stays quiet. Admit the false ALERT on col_d_lower at month 4 (settling peer drags the seasonal estimate; clears later).
4:20 Orders and record: printable inspection order, "KEYSTONE recommends, the custodian decides", archive with integrity check.
4:45 Close: proven = pipeline end to end on simulated data. Not proven = real-world accuracy. Next = one real capture with printed scale markers to test the 1 to 3 cm claim.

Do not say: tested on a real monument; centimetre accuracy achieved; ready for deployment.
