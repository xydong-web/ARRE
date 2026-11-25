"""Reader-facing names; archived experimental tags remain unchanged."""

ABLATIONS = (
    ("r2E4L4", "w/o Prefix Reuse", "Without RHC"),
    ("r4CP4", "w/o Critic", "Without RFG"),
    ("r4CE4", "w/o Synthesis Prior", "Without SPP and look-ahead"),
    ("r4CE4S4", "w/o Look-ahead", "Without look-ahead"),
)


def ablation_names(text):
    for _, old, new in ABLATIONS:
        text = text.replace(old, new)
    return text
