Humming 0.1.13, not a text transform.

vLLM 0.27.1 pins Humming 0.1.10, whose SM121 selector falls back to SM89 and
whose NVML bandwidth probe fails on GB10. 0.1.13 adds a native SM121 heuristic
and a GB10 bandwidth fallback. The profile serves with `--linear-backend
humming`, so this is a build dependency of the composed image rather than an
optional layer, and the Dockerfile installs and verifies it in place.
