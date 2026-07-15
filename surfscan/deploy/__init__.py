"""Deploy cost-model — analytical deployability (measured now, projected onto accelerators).

Per-model compute footprint (params, inference FLOPs, peak activation memory, disk fp32/int8) is
measured here; the projection onto candidate edge accelerators (fits-memory / roofline-latency band /
utilization %) is the honest boundary — a projection with an efficiency band, NOT a measured FPS.
"""
