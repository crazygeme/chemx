import unittest

from chemx.core.coding import ActionKind, parse_action


class ActionParsingTests(unittest.TestCase):
    def test_parses_action_wrapped_in_explanatory_text(self) -> None:
        action = parse_action(
            'I will read the file next.\n'
            '{"kind":"read_file","path":"src/example.py"}\n'
            "Then I will inspect the result."
        )

        self.assertEqual(action.kind, ActionKind.READ_FILE)
        self.assertEqual(action.path, "src/example.py")

    def test_parses_fenced_action_with_surrounding_text(self) -> None:
        action = parse_action(
            "Here is the next action:\n"
            "```json\n"
            '{"kind":"run_command","command":["gcc","hello.c"]}\n'
            "```"
        )

        self.assertEqual(action.kind, ActionKind.RUN_COMMAND)
        self.assertEqual(action.command, ("gcc", "hello.c"))

    def test_rejects_multiple_valid_actions(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one JSON action"):
            parse_action(
                '{"kind":"list_files"}\n'
                '{"kind":"finish","message":"done"}'
            )

    def test_rejects_text_without_action_object(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid JSON object"):
            parse_action("I will inspect the repository now.")

    def test_unsupported_kind_reports_received_and_supported_values(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "received 'scan_directory'.*list_files",
        ):
            parse_action('{"kind":"scan_directory","path":"src"}')


if __name__ == "__main__":
    unittest.main()
