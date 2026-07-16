"""The control/VLA leg — a policy-learning application on the shared `core` kernel, sibling to `surfscan`.

Where `surfscan` is the perception leg (detect anomalies, score AU-PRO), `control` is the action leg: learn
a policy in simulation and measure how much it loses when the dynamics shift — the sim-to-real **policy
gap**, the leg's headline number. It speaks the `core.policy` / `core.rollout` contracts; it imports no
`surfscan` (import-linter enforces the sibling independence).
"""
