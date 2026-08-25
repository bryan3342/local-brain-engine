"""Deterministic half of the ETL: extract, filter, queue, reconcile, learn.

Nothing in this package makes a judgement call. Rules only, so every output is
reproducible byte-for-byte. Judgement lives in the `brain-etl` skill.
"""
