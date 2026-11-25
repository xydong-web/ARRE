"""SPP (Synthesis-Planning Prior) is the frozen RFG/SPP route-family prior.

This is a part of the RFG/SPP behaviour sampler, not a separate learned
generator or external inference model.
"""
SPP_IMPLEMENTATION = 'cgflow.lanes.rfg_spp:RFGSPPState'
SPP_CONFIG_KEYS = ('rfg_spp.syn_prior','rfg_spp.syn_kappa',
                   'rfg_spp.syn_lookahead')
SPP_FROZEN_FILE = 'code/frozen_prior/syn_prior_v1.frozen.json'
