# -*- coding: UTF-8 -*-
"""
TIGERRunner

Minimal runner for TIGER models. Currently delegates behavior to BaseRunner but
keeps a separate class so users can select `--runner TIGERRunner` if desired.
"""
from helpers.BaseRunner import BaseRunner


class TIGERRunner(BaseRunner):
    @staticmethod
    def parse_runner_args(parser):
        # reuse BaseRunner args; could extend later with TIGER-specific options
        return BaseRunner.parse_runner_args(parser)

    # All behavior inherited from BaseRunner; override hooks if needed later
