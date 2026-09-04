# results/

Committed measurement data. Keep every CSV a run produces, including partial
ones from jobs that hit the time limit — the sweep flushes per row, so a killed
job still holds every point it measured.

Name files `bench_speed_<cluster>_<gpu>_<jobid>.csv` so a later reader can tell
an A100 run from an H100 run without opening them. The GPU model is also the
first column of every row.
