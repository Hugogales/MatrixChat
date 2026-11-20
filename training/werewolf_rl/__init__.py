"""Werewolf Stage-B RL: shared game engine, rewards, critic and PPO.

See ``WEREWOLF_RL_PLAN.md`` for the design. ``game`` plays one game in the
shared KV-cached matrix (used by ``scripts/eval/werewolf_game.py`` and the
trainer), ``rewards`` scores it (Pillars 1-3), ``critic`` holds the value
model and per-role advantage normalizer, and ``ppo`` turns recorded games
into losses.
"""