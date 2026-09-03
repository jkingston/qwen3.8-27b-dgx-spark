# Contributing

Contributions are welcome. This project is about squeezing a dense 27B model
into useful speeds on one DGX Spark, so measurements are as valuable as code.

## How to contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Commit your changes
4. Push to the branch (`git push origin my-feature`)
5. Open a pull request

## What is especially useful

- Benchmarks on DGX Spark, including negative results — an optimization that
  does nothing is worth documenting so nobody repeats it
- Kernel work on the decode path: the LM head, the linear-attention layers,
  the INT4 GEMM path
- Speculative decoding tuning: acceptance rates, draft-length sweeps
- Quantization recipes that trade quality for bandwidth in a measurable way
- Fixes to the installer and launch scripts
- Documentation, especially anything that shortens the path from clone to a
  running server

## Guidelines

- Keep pull requests focused — one feature or fix per pull request
- Include benchmark numbers when a change touches performance, and say how
  they were measured: prompt set, concurrency, and how many runs
- State what you did *not* verify. A stated gap is more useful than an
  implied guarantee
- Patches against vLLM are version-specific; name the version you tested
- Follow the existing code style

## License

By contributing, you agree that your contributions will be licensed under the
[MIT License](LICENSE).
