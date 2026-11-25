"""Compositions of the innovation lanes (integration branch only).  Every lane keeps its own config and default-OFF switches, so a composed trainer with
all switches off is the baseline.  Select with  run_opt.py --trainer cgflow.lanes.combo:<Class>  and the usual  --set lane_x.enable=true  switches.

  VinaMOOTrainer_RFGSPP_with_replay   = E (sampler side: critic-tilted behaviour policy)  +  B (train side: prioritised replay + hindsight + UTD)
  VinaMOOTrainer_RFGSPP_with_pose_proxy   = E  +  D (data side: oversample -> pose-proxy gate -> oracle)
  VinaMOOTrainer_RFGSPP_with_replay_and_pose_proxy  = E  +  B  +  D

MRO is  composed -> E -> B -> D -> VinaMOOTrainer_semla ;  E.setup/setup_algo/setup_model, B.setup/train_batch and D.build_training_data_loader
are cooperative (each calls super()), and D's loader registers every entry of self.sampling_hooks (E's CriticEngine, B's ReplayEngine).
Cull-safe resume: the composed trainers carry the RFG/SPP critic state exactly; the lane-B replay buffer holds references to sampler objects (poses, graphs, ~19 GB RSS)
and is NOT checkpointed - after a resume it refills within a few iterations (documented approximation; B's RNG streams are restored)."""
from __future__ import annotations

from cgflow.lanes.lane_b import VinaMOOTrainer_laneB
from cgflow.lanes.lane_d import VinaMOOTrainer_laneD
from cgflow.lanes.rfg_spp import VinaMOOTrainer_RFGSPP


class _ComboResume:
    """rs_extra_state / rs_load_extra_state that merge the per-lane states."""

    def rs_extra_state(self):
        out = VinaMOOTrainer_RFGSPP.rs_extra_state(self)
        eng = getattr(self, "_lane_b_engine", None)
        if eng is not None:
            out["lane_b_light"] = {"rng": eng.rng.bit_generator.state, "gen": self._lane_b_gen.get_state(), "current_iter": int(eng.current_iter),
                                   "n_inserted": int(eng.n_inserted), "nonfinite": int(self._lane_b_nonfinite)}
        return out

    def rs_load_extra_state(self, ex):
        VinaMOOTrainer_RFGSPP.rs_load_extra_state(self, ex)
        d = ex.get("lane_b_light")
        eng = getattr(self, "_lane_b_engine", None)
        if d is not None and eng is not None:
            eng.rng.bit_generator.state = d["rng"]
            self._lane_b_gen.set_state(d["gen"])
            eng.current_iter = int(d["current_iter"])
            self._lane_b_nonfinite = int(d["nonfinite"])


class VinaMOOTrainer_RFGSPP_with_replay(_ComboResume, VinaMOOTrainer_RFGSPP, VinaMOOTrainer_laneB):
    pass


class VinaMOOTrainer_RFGSPP_with_pose_proxy(_ComboResume, VinaMOOTrainer_RFGSPP, VinaMOOTrainer_laneD):
    pass


class VinaMOOTrainer_RFGSPP_with_replay_and_pose_proxy(_ComboResume, VinaMOOTrainer_RFGSPP, VinaMOOTrainer_laneB, VinaMOOTrainer_laneD):
    pass


# ------------------------------------------------------------------------------------------------------------------------------------------------
# C (elite-prefix resampling, sampler-side forced actions + portfolio data source) on top of E (critic tilt of the behaviour logits)
#   sampler MRO: C._sample_action -> E._sample_action -> baseline: the tilt acts on every step, the forced prefix actions overwrite the sampled ones afterwards.
#   Cull-safe (since RHC commit c8f5d50): _ARREResume merges RFG/SPP's critic state, lane B's light state and RHC's elite archive / bandit / kernel-Z / score cache.
# ------------------------------------------------------------------------------------------------------------------------------------------------
from cgflow.algo.semla_algo import SynthesisTB3D_semla  # noqa: E402
from cgflow.lanes.rhc import SyntheticPathSampler3D_semlaRHC, VinaMOOTrainer_RHC  # noqa: E402
from cgflow.lanes.rfg_spp import SyntheticPathSampler3D_semlaRFGSPP  # noqa: E402


class SyntheticPathSampler3D_semlaARRE(SyntheticPathSampler3D_semlaRHC, SyntheticPathSampler3D_semlaRFGSPP):
    pass


class SynthesisTB3D_semlaARRE(SynthesisTB3D_semla):
    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler3D_semlaARRE(
            self.ctx,
            self.env,
            self.action_subsampler,
            max_len=self.max_len,
            importance_temp=self.importance_temp,
            sample_temp=self.sample_temp,
            correct_idempotent=self.cfg.do_correct_idempotent,
            pad_with_terminal_state=self.cfg.do_parameterize_p_b,
            num_workers=self.global_cfg.num_workers_retrosynthesis,
        )


class _ARREResume(_ComboResume):
    """_ComboResume (E + light B state) plus RHC's state (stored under the key 'rhc'; RHC's trainer attribute is rhc_state, RFG/SPP's is lane_state)."""

    def rs_extra_state(self):
        out = _ComboResume.rs_extra_state(self)
        out.update(VinaMOOTrainer_RHC.rs_extra_state(self))
        return out

    def rs_load_extra_state(self, ex):
        _ComboResume.rs_load_extra_state(self, ex)
        VinaMOOTrainer_RHC.rs_load_extra_state(self, ex)


class VinaMOOTrainer_ARRE(_ARREResume, VinaMOOTrainer_RFGSPP, VinaMOOTrainer_RHC):

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D_semlaARRE(self.env, self.ctx, self.cfg)

    def build_training_data_loader(self):
        loader = super().build_training_data_loader()  # RHC's loader (creates rhc_state); RFG/SPP's state exists since setup()
        c, e = getattr(self, "rhc_state", None), getattr(self, "lane_state", None)
        if c is not None and c.cfg.route_aware:
            assert e is not None and e.syn is not None, "rhc.route_aware needs RFG/SPP with a route prior (rfg_spp.enable, syn_prior, syn_kappa > 0)"
            c.route_logw = e.prefix_logw  # elite prefixes are proposed with the same route potential as the behaviour policy
        return loader


class VinaMOOTrainer_ARRE_with_replay(VinaMOOTrainer_laneB, VinaMOOTrainer_ARRE):
    pass  # resume state comes from _ARREResume (MRO: BCE, B, CE, _ARREResume, _ComboResume, E, C, ...)
