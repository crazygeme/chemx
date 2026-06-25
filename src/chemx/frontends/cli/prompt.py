"""Modern terminal input for interactive frontend sessions."""

from dataclasses import dataclass, field
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


# Modern terminals using CSI-u or modifyOtherKeys can distinguish Shift+Enter.
# Normalize both common encodings to the same chord as Alt+Enter.
ANSI_SEQUENCES["\x1b[13;2u"] = (Keys.Escape, Keys.ControlM)
ANSI_SEQUENCES["\x1b[27;2;13~"] = (Keys.Escape, Keys.ControlM)


def _create_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add(Keys.ControlM)
    def _submit(event: Any) -> None:
        event.current_buffer.validate_and_handle()

    @bindings.add(Keys.Escape, Keys.ControlM)
    def _insert_newline(event: Any) -> None:
        event.current_buffer.insert_text("\n")

    return bindings


@dataclass
class InteractivePrompt:
    """Read editable, multiline input with history and familiar shortcuts."""

    session: PromptSession[str] = field(
        default_factory=lambda: PromptSession(
            history=InMemoryHistory(),
            multiline=True,
            key_bindings=_create_key_bindings(),
            enable_open_in_editor=False,
        )
    )

    def read(self) -> str:
        """Read one submitted user message."""
        return self.session.prompt(
            [("class:prompt", "you> ")],
            prompt_continuation=lambda width, line_number, is_soft_wrap: (
                " " * max(0, width - 2) + "· "
            ),
        )
