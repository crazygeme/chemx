import unittest

from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.keys import Keys

from chemx.frontends.cli.prompt import _create_key_bindings


class InteractivePromptTests(unittest.TestCase):
    def test_enter_submits_and_alt_enter_inserts_newline(self) -> None:
        bindings = _create_key_bindings()

        self.assertTrue(bindings.get_bindings_for_keys((Keys.ControlM,)))
        self.assertTrue(
            bindings.get_bindings_for_keys((Keys.Escape, Keys.ControlM))
        )

    def test_modern_shift_enter_sequences_map_to_newline_chord(self) -> None:
        expected = (Keys.Escape, Keys.ControlM)

        self.assertEqual(ANSI_SEQUENCES["\x1b[13;2u"], expected)
        self.assertEqual(ANSI_SEQUENCES["\x1b[27;2;13~"], expected)


if __name__ == "__main__":
    unittest.main()
