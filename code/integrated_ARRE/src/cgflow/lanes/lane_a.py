"""Lane A - PAN-Sub trainer.

    run/run_opt.py --trainer cgflow.lanes.lane_a:VinaMOOTrainer_laneA \
        --set lane_a.enable=true --set lane_a.normalizer=pan --set lane_a.behavior=full

With `lane_a.enable=false` (the default) this trainer is the baseline
`cgflow.tasks.semla_vina:VinaMOOTrainer_semla`: no block table is built, no cache is
allocated, and `PANRxnFlow._make_cat` returns the baseline `RxnActionCategorical`.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch_geometric.data as gd

from rxnflow.base.trainer import mogfn_trainer
from cgflow.config import Config
from cgflow.tasks.semla_vina import VinaTrainer_semla, VinaMOOTask_semla
from cgflow.lanes.pansub.model import PANRxnFlow
from cgflow.lanes.pansub.pan import PanEngine
from cgflow.lanes.pansub.algo import SynthesisTB3D_laneA


class VinaTrainer_laneA(VinaTrainer_semla):
    pan_engine: PanEngine | None

    def setup_algo(self):
        assert self.cfg.algo.method == "TB"
        self.algo = SynthesisTB3D_laneA(self.env, self.ctx, self.cfg)

    def setup_model(self):
        self.model = PANRxnFlow(
            self.ctx,
            self.cfg,
            do_bck=self.cfg.algo.tb.do_parameterize_p_b,
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
        )

    def setup(self):
        super().setup()
        self.pan_engine = None
        cfg_a = self.cfg.lane_a
        if not cfg_a.enable:
            print("[lane_a] disabled -> baseline behaviour", flush=True)
            self.model.attach_pan(None, "train")
            if self.sampling_model is not self.model:
                self.sampling_model.attach_pan(None, "sample")
            return
        assert cfg_a.normalizer in ("uniform", "pan"), cfg_a.normalizer
        assert cfg_a.behavior in ("subsample", "full"), cfg_a.behavior
        t0 = time.time()
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        engine = PanEngine(
            cfg_a,
            self.ctx,
            self.device,
            num_emb_block=self.cfg.model.num_emb_block,
            seed=self.cfg.seed,
        )
        self.pan_engine = engine
        # The proposal / behaviour policy is always built from the EMA *sampling* model: one
        # cache (1.3 GB in fp16) instead of two, and the sampler is the policy that actually
        # generates the data.  Importance sampling stays unbiased for the training model.
        engine.cache.refresh(self.sampling_model, 0)
        self.model.attach_pan(engine, "train")
        if self.sampling_model is not self.model:
            self.sampling_model.attach_pan(engine, "sample")
        nb = engine.table.nbytes()
        print(
            f"[lane_a] PAN enabled: normalizer={cfg_a.normalizer} behavior={cfg_a.behavior} "
            f"K={cfg_a.num_samples} lambda={cfg_a.defensive_mix} eps={cfg_a.behavior_eps} "
            f"refresh_every={cfg_a.refresh_every}",
            flush=True,
        )
        print(
            f"[lane_a] block table: {engine.table.num_blocks} blocks, "
            f"fp_packed {nb['fp_packed'] / 2**30:.2f} GiB on {engine.table.table_device}, "
            f"cache {engine.cache.emb.numel() * engine.cache.emb.element_size() / 2**30:.2f} GiB on "
            f"{engine.cache.emb.device}, build+refresh {time.time() - t0:.1f}s "
            f"(first refresh {engine.cache.last_refresh_seconds:.2f}s)",
            flush=True,
        )

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> dict[str, Any]:
        engine = getattr(self, "pan_engine", None)
        if engine is not None:
            engine.iteration = train_it
            engine._timers.clear()
            engine.stats.clear()
        info = super().train_batch(batch, epoch_idx, batch_idx, train_it)
        try:  # resident set size of THIS process (sacct's MaxRSS sums the forked docking pool)
            import psutil

            info["lane_a/rss_GiB"] = psutil.Process().memory_info().rss / 2**30
        except Exception:
            pass
        if engine is not None:
            t_pan = engine._timers.get("pan_pass", 0.0)
            info.update(engine.stats)
            info["lane_a/pan_seconds"] = t_pan
            every = max(1, int(self.cfg.lane_a.refresh_every))
            if train_it % every == 0:
                dt = engine.cache.refresh(self.sampling_model, train_it)
                info["lane_a/refresh_seconds"] = dt
                info["lane_a/cache_absmax"] = engine.cache.last_absmax
            if torch.cuda.is_available():
                info["lane_a/gpu_peak_GiB"] = torch.cuda.max_memory_allocated(self.device) / 2**30
        return info


@mogfn_trainer
class VinaMOOTrainer_laneA(VinaTrainer_laneA):
    task: VinaMOOTask_semla

    def set_default_hps(self, base: Config):
        super().set_default_hps(base)
        base.task.moo.objectives = ["vina", "qed"]

    def setup_task(self):
        self.task = VinaMOOTask_semla(cfg=self.cfg, wrap_model=self._wrap_for_mp)

    def add_extra_info(self, info):
        for prop, fr in self.task.avg_reward_info.items():
            info[f"sample_r_{prop}_avg"] = fr
        super().add_extra_info(info)
