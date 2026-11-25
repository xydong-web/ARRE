"""Read older ARRE run settings and checkpoint metadata in the public runners."""

_PREVIOUS = "lane"
PREVIOUS_RHC = f"{_PREVIOUS}_c"
COMPONENT_NAMES = {PREVIOUS_RHC: "rhc", f"{_PREVIOUS}_e": "rfg_spp"}
_TRAINER_SUFFIXES = {
    "C": "RHC",
    "E": "RFGSPP",
    "CE": "ARRE",
    "BE": "RFGSPP_with_replay",
    "DE": "RFGSPP_with_pose_proxy",
    "BDE": "RFGSPP_with_replay_and_pose_proxy",
    "BCE": "ARRE_with_replay",
}


def trainer_path(path):
    module, cls = path.split(":", 1)
    package, _, name = module.rpartition(".")
    for previous, current in COMPONENT_NAMES.items():
        if name == previous or name.startswith(previous + "_"):
            module = f"{package}.{current}{name[len(previous):]}"
            break
    prefix, sep, suffix = cls.rpartition("_" + _PREVIOUS)
    if sep and suffix in _TRAINER_SUFFIXES:
        cls = f"{prefix}_{_TRAINER_SUFFIXES[suffix]}"
    return module, cls


def checkpoint_extra(extra):
    """Normalize component state, including metric keys saved in RHC state."""
    updated = dict(extra)
    for previous, current in COMPONENT_NAMES.items():
        if previous in updated:
            if current in updated:
                raise ValueError(f"Checkpoint contains two states for {current}")
            updated[current] = updated.pop(previous)
    rhc = updated.get("rhc")
    if rhc is not None and "cond_stats" in rhc.get("state", {}):
        state = dict(rhc["state"])
        previous_prefix = PREVIOUS_RHC + "/"
        state["cond_stats"] = {
            "rhc/" + key[len(previous_prefix):] if key.startswith(previous_prefix) else key: value
            for key, value in state["cond_stats"].items()
        }
        updated["rhc"] = dict(rhc, state=state)
    return updated
