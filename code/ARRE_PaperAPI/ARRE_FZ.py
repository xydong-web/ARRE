"""FZ is an experimental arm: freeze both generator and log-Z optimizer.

Archived x4nw2F vs x4nw2FZ YAML files are authoritative; do not reinterpret
FZ as a fourth learned model component.
"""
FZ_EXTRA_SWITCHES = ('opt.learning_rate=0.0', 'algo.Z_learning_rate=0.0')
