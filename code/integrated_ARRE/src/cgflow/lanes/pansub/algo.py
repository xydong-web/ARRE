"""Lane A / PAN-Sub: sampler + algorithm wiring."""

from __future__ import annotations

from gflownet.envs.graph_building_env import ActionIndex
from torch_geometric import data as gd

from rxnflow.policy.action_categorical import RxnActionCategorical
from cgflow.algo.semla_algo import SynthesisTB3D_semla, SyntheticPathSampler3D_semla

from .categorical import PANActionCategorical


class SyntheticPathSampler3D_laneA(SyntheticPathSampler3D_semla):
    """Same trajectory sampler as the baseline, but with A2 the action is drawn from the FULL
    legal action set instead of from the random 1% sub-sample."""

    def _sample_action(
        self,
        torch_graphs: list[gd.Data],
        fwd_cat: RxnActionCategorical,
        random_action_prob: float = 0,
    ) -> list[ActionIndex]:
        if isinstance(fwd_cat, PANActionCategorical) and fwd_cat.use_full_behavior:
            return fwd_cat.sample_full(random_action_prob)
        return super()._sample_action(torch_graphs, fwd_cat, random_action_prob)


class SynthesisTB3D_laneA(SynthesisTB3D_semla):
    def setup_graph_sampler(self):
        self.graph_sampler = SyntheticPathSampler3D_laneA(
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
