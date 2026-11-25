"""Innovation lanes.  Each lane ships a Config dataclass and a trainer subclass; every switch defaults to OFF
so that importing / selecting a lane trainer without its flags reproduces the baseline exactly."""

"""Innovation lanes for CGFlow.
Each lane ships a trainer subclass selected on the command line with
``run_opt.py --trainer cgflow.lanes.lane_<x>:VinaMOOTrainer_lane<X>`` plus a
config dataclass on :class:`cgflow.config.Config` whose switches all default to
OFF, so an un-switched lane reduces exactly to the paper baseline.
"""
