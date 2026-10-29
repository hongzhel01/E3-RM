# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Modified for E3-RM: namespace migration and maintenance comments (2026-10-28).
# 统一 e3-rm 命令入口，将训练子命令交给对应启动器。
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

import sys
from argparse import ArgumentParser

from .commands import get_commands
from .commands.base import Command, CommandContext


def _build_parser(commands: list[Command]) -> ArgumentParser:
    parser = ArgumentParser(prog="e3-rm", allow_abbrev=False)
    subparsers = parser.add_subparsers(help="available commands", dest="command")

    for command in commands:
        command.register(subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the E3-RM CLI."""
    commands = get_commands()
    commands_by_name = {command.name: command for command in commands}
    parser = _build_parser(commands)
    argv = list(sys.argv[1:] if argv is None else argv)

    args, _ = parser.parse_known_args(argv)
    command_name = getattr(args, "command", None)
    if command_name is None:
        parser.print_help()
        return 0

    command = commands_by_name[command_name]
    context = CommandContext(argv=argv)
    return command.run(args, context)


if __name__ == "__main__":
    raise SystemExit(main())
