"""RFG (Reactant Feedback Guidance) is the RFG/SPP Bayesian-ridge critic.

RFG-specific configuration disables the route-prior knobs but retains the
structure-preserving group normalization of the reactant tilt.
"""
RFG_IMPLEMENTATION = 'cgflow.lanes.rfg_spp:BlockCritic'
RFG_CONFIG_KEYS = ('rfg_spp.enable','rfg_spp.tilt','rfg_spp.group_norm',
                   'rfg_spp.ridge_grid','rfg_spp.kl_target')
