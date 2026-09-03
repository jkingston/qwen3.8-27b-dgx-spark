# Security Policy

## Reporting a vulnerability

Report security issues by opening a GitHub issue. If the issue is sensitive,
write to `albond.dev@proton.me` instead.

This project ships build files, patch scripts, and launch scripts for local
inference. It has no network-facing service of its own beyond the vLLM API
server it starts on the machine you run it on. Realistic concerns are:

- Dockerfile misconfiguration
- Unsafe defaults in launch scripts, in particular binding the API server to a
  public interface without authentication
- Dependency vulnerabilities in the base images and pinned wheels
- Patch scripts editing files outside the intended vLLM installation

## Scope

The launch scripts bind the API server to the loopback interface.
`DENSESPARK_BIND_HOST` publishes it on another interface, and `0.0.0.0` on all
of them. Exposing it is your decision and your responsibility: vLLM's
OpenAI-compatible server has no authentication of its own, so anything that can
reach the port can use the model and read every prompt sent to it.
