# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Modified for E3-RM: namespace migration and maintenance comments (2026-10-28).
# 保留本模块原有行为，配合 E3-RM 包入口使用；关键依赖按需加载。
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from argparse import Namespace

from .base import Command, CommandContext


class EnvCommand(Command):
    """CLI command that prints E3-RM environment information."""

    def __init__(self):
        super().__init__(name="env", help_text="Print the environment information")

    def register(self, subparsers) -> None:
        subparsers.add_parser(self.name, help=self.help_text)

    def run(self, args: Namespace, context: CommandContext) -> int:
        from ...scripts.env import print_env

        print_env()
        return 0
