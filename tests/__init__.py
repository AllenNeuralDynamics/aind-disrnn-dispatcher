"""Dispatcher test suite.

Covers the launcher helpers -- the pure string/dict/path functions every launch
goes through. Deliberately free of SLURM, Beaker, W&B and network access, so
the suite runs anywhere in seconds. See #97.
"""
