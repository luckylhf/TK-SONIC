# SPDX-License-Identifier: AGPL-3.0-or-later
# Distributed as part of the SONIC teleoperation GUI; see README-LICENSE.md.
# Licensing metadata clarified by luckylhf on 2026-08-18.

import subprocess

import click


@click.command()
def teleop():
    """CLI for interacting with the osmo."""
    subprocess.run(["python", "decoupled_wbc/control/teleop/gui/main.py"])


@click.group()
def cli():
    """CLI for interacting with the osmo."""


cli.add_command(teleop)

if __name__ == "__main__":
    cli()
