"""Memory watchdog tests without host inspection or Docker calls."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    'memory_watch', Path(__file__).resolve().parents[1] / 'scripts/watch_model_memory.py')
watch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


class MemoryWatchTests(unittest.TestCase):
    def test_absolute_floors(self):
        self.assertFalse(watch.below_floor({'available_kb': 768 * 1024, 'free_kb': 512 * 1024}))
        self.assertTrue(watch.below_floor({'available_kb': 768 * 1024 - 1, 'free_kb': 512 * 1024}))
        self.assertTrue(watch.below_floor({'available_kb': 768 * 1024, 'free_kb': 512 * 1024 - 1}))

    def test_proc_snapshot(self):
        with patch.object(watch.Path, 'read_text', side_effect=[
                'MemAvailable: 900000 kB\nMemFree: 800000 kB\n',
                'some avg10=1.0 avg60=0.0 total=2\nfull avg10=0.5 avg60=0.0 total=1\n']):
            sample = watch.snapshot()
        self.assertEqual(sample['available_kb'], 900000)
        self.assertEqual(sample['free_kb'], 800000)
        self.assertEqual(sample['full_psi10'], 0.5)

    def test_stops_only_named_container(self):
        with patch('sys.argv', ['watch', '--container', 'glm53-exl3-head']), patch.object(
                watch, 'snapshot', return_value={'available_kb': 1, 'free_kb': 1}), patch.object(
                watch.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(watch.main(), 1)
        run.assert_called_once_with(
            ['docker', 'stop', '-t', '2', 'glm53-exl3-head'], timeout=20, check=True)


if __name__ == '__main__':
    unittest.main()
