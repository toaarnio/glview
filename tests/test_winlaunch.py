import tempfile
import unittest
from pathlib import Path

from glview import winlaunch


class _FakeSpawnBackend:
    def __init__(self):
        self.calls = []

    def popen(self, args):
        self.calls.append(args)


class _FakeTimeBackend:
    def __init__(self):
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)


class WinLaunchTests(unittest.TestCase):

    def test_launch_leader_batches_multiple_event_files_into_one_launch(self):
        spawn = _FakeSpawnBackend()
        timer = _FakeTimeBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            spool_root = Path(tmpdir)
            spool_dir = winlaunch._spool_dir(spool_root)
            spool_dir.mkdir(parents=True, exist_ok=True)
            (spool_dir / "000-manual.json").write_text('["b.png", "c.png"]', encoding="utf-8")

            handled = winlaunch.launch(
                ["d.png"],
                spool_root=spool_root,
                spawn_backend=spawn,
                time_backend=timer,
            )

        self.assertTrue(handled)
        self.assertEqual(timer.sleeps, [winlaunch.DEBOUNCE_SECONDS])
        self.assertEqual(len(spawn.calls), 1)
        self.assertEqual(spawn.calls[0][-3:], ["b.png", "c.png", "d.png"])

    def test_launch_follower_only_queues_file(self):
        spawn = _FakeSpawnBackend()
        timer = _FakeTimeBackend()

        with tempfile.TemporaryDirectory() as tmpdir:
            spool_root = Path(tmpdir)
            spool_dir = winlaunch._spool_dir(spool_root)
            spool_dir.mkdir(parents=True, exist_ok=True)
            (spool_dir / winlaunch.LOCK_FILENAME).write_text("", encoding="utf-8")

            handled = winlaunch.launch(
                ["a.png"],
                spool_root=spool_root,
                spawn_backend=spawn,
                time_backend=timer,
            )

            event_files = list(spool_dir.glob("*.json"))

        self.assertTrue(handled)
        self.assertEqual(timer.sleeps, [])
        self.assertEqual(spawn.calls, [])
        self.assertEqual(len(event_files), 1)

    def test_viewer_command_uses_module_entrypoint_for_python_run(self):
        command = winlaunch._viewer_command(["a.png", "b.png"])

        self.assertEqual(command[1:3], ["-m", "glview.glview"])
        self.assertEqual(command[-2:], ["a.png", "b.png"])


if __name__ == "__main__":
    unittest.main()
