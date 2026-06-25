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

    def test_list_tool_can_be_scoped_to_one_directory(self) -> None:
        writer = WriteTool(self.paths)
        writer.create("src/chemx/example.py", "value = 1\n")
        writer.create(".venv/bin/activate", "generated\n")

        files = ListTool(self.paths).run("src/chemx")

        self.assertEqual(files, ["src/chemx/example.py"])

    def test_list_tool_honors_nested_gitignore_rules(self) -> None:
        writer = WriteTool(self.paths)
        writer.create(".gitignore", ".venv/\n*.log\n!important.log\n")
        writer.create("src/.gitignore", "generated/\n")
        writer.create("src/example.py", "value = 1\n")
        writer.create("src/debug.log", "ignored\n")
        writer.create("src/important.log", "included\n")
        writer.create("src/generated/output.py", "ignored\n")
        writer.create(".venv/bin/activate", "ignored\n")

        files = ListTool(self.paths).run()

        self.assertEqual(
            files,
            [
                ".gitignore",
                "src/.gitignore",
                "src/example.py",
                "src/important.log",
            ],
        )

    def test_scoped_list_tool_still_honors_root_gitignore(self) -> None:
        writer = WriteTool(self.paths)
        writer.create(".gitignore", "src/chemx/private.py\n")
        writer.create("src/chemx/public.py", "value = 1\n")
        writer.create("src/chemx/private.py", "secret = 1\n")

        files = ListTool(self.paths).run("src/chemx")

        self.assertEqual(files, ["src/chemx/public.py"])

    def test_list_tool_supports_anchored_and_recursive_patterns(self) -> None:
        writer = WriteTool(self.paths)
        writer.create(
            ".gitignore",
            "/root-only.txt\n**/cache/*.bin\nbuild/file[0-9].tmp\n",
        )
        writer.create("root-only.txt", "ignored\n")
        writer.create("nested/root-only.txt", "included\n")
        writer.create("src/cache/data.bin", "ignored\n")
        writer.create("build/file1.tmp", "ignored\n")
        writer.create("build/filex.tmp", "included\n")

        files = ListTool(self.paths).run()

        self.assertEqual(
            files,
            [
                ".gitignore",
                "build/filex.tmp",
                "nested/root-only.txt",
            ],
        )

    def test_search_tool_finds_text(self) -> None:
        WriteTool(self.paths).create("example.txt", "needle\n")

        result = SearchTool(self.paths).run("needle")

        self.assertIn("example.txt", result)

    def test_bash_tool_is_disabled_by_default(self) -> None:
        with self.assertRaisesRegex(ValueError, "disabled"):
            BashTool(self.paths).run("echo unsafe")


if __name__ == "__main__":
    unittest.main()
