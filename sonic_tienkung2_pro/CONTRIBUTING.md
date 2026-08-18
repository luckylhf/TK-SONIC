# Contributing to SONIC TienKung2 Pro

We welcome contributions from the community! Here's how to get started.

## Reporting Issues

- Search the issue tracker supplied by this package's distributor first
- Open a new issue there with a clear description, error messages, and steps to reproduce
- Include your Python version, OS, GPU, and Isaac Lab version

## Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b my-feature`)
3. Make your changes
4. Run the pre-flight check: `python check_environment.py`
5. Commit and push to your fork
6. Open a pull request through the repository or contribution channel supplied by
   this package's distributor

### Guidelines

- Keep PRs focused on a single change
- Follow existing code style (no linter is enforced, but be consistent)
- Update documentation if your change affects user-facing behavior
- Add yourself to the PR description if you'd like credit

## Development Setup

See this package's `README.md` and local documentation for the training environment.

## Questions

For questions, use the support or issue channel supplied by this package's
distributor. NVIDIA upstream contacts are not responsible for downstream changes.

## License

Contributions outside `decoupled_wbc/control/teleop/gui/` are accepted under the
Apache License 2.0 unless a component notice states otherwise. Contributions to
that GUI are accepted under AGPL-3.0-or-later. See [LICENSE](LICENSE) and
[NOTICE](NOTICE) before contributing.
