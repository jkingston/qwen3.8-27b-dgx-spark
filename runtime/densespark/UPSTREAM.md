# Vendored upstream

This directory vendors
[`albond/DenseSpark-Qwen3.8-27B`](https://github.com/albond/DenseSpark-Qwen3.8-27B)
release 1.3 at commit `9ae122f757bbae28c875d00c8b186c4837187434`.

It is vendored rather than added as a submodule so a checkout of this deployment
repository is complete and installable. Local changes are limited to integration
fixes and should be kept small enough to reapply when updating the upstream pin.

Integration changes:

- `bench_serving.py` has separate host-visible and container-visible base URLs,
  allowing a server published on a non-default host port to be benchmarked.
