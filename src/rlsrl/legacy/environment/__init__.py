"""Legacy environments are registered safely.
"""
from rlsrl.api.environment import register

register("atari", "AtariEnvironment", "rlsrl.legacy.environment.atari.atari_env")
