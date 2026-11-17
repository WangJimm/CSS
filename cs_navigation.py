"""Compatibility entry point for :mod:`scripts.run_experiment`."""

from scripts.run_experiment import AGENT_CHOICES, MODE_CHOICES, main, parse_args


if __name__ == "__main__":
    raise SystemExit(main())
