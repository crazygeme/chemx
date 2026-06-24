import tempfile
import unittest
from pathlib import Path

from chemx.tools import (
    BashTool,
    EditTool,
    ListTool,
    ReadTool,
    SearchTool,
    WorkspacePaths,
    WriteTool,
)


class CommonToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.paths = WorkspacePaths(self.root)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_file_tools_are_workspace_scoped(self) -> None:
        writer = WriteTool(self.paths)
        writer.create("src/example.py", "value = 1\n")

        self.assertEqual(ListTool(self.paths).run(), ["src/example.py"])
        self.assertEqual(ReadTool(self.paths).run("src/example.py"), "value = 1\n")

        EditTool(self.paths).replace_exact("src/example.py", "1", "2")
        self.assertEqual(ReadTool(self.paths).run("src/example.py"), "value = 2\n")

        with self.assertRaisesRegex(ValueError, "escapes workspace"):
            ReadTool(self.paths).run("../outside.txt")

    def test_search_tool_finds_text(self) -> None:
        WriteTool(self.paths).create("example.txt", "needle\n")

        result = SearchTool(self.paths).run("needle")

        self.assertIn("example.txt", result)

    def test_bash_tool_is_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "disabled"):
            BashTool(self.paths).run("echo unsafe")


if __name__ == "__main__":
    unittest.main()
