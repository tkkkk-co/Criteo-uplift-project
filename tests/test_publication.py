"""Snapshot routing and real small-model demo; never depend on raw Criteo."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import unittest
import numpy as np
from criteo_uplift.results import result_directory
from criteo_uplift.runtime import configure_runtime

configure_runtime()
from criteo_uplift.demo import run_demo


class PublicationTests(unittest.TestCase):
    def test_local_priority_and_explicit_snapshot_mode(self):
        with TemporaryDirectory() as d:
            root = Path(d)
            local, public = root / 'artifacts/run', root / 'reports/published/run'
            public.mkdir(parents=True)
            with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'auto'}):
                self.assertEqual(result_directory('run', root), public)
                local.mkdir(parents=True)
                self.assertEqual(result_directory('run', root), local)
            with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'published'}):
                self.assertEqual(result_directory('run', root), public)

    def test_invalid_paths_and_modes(self):
        for name in ['', '..', '../data', '/tmp/test', 'a/b']:
            with self.assertRaises(ValueError):
                result_directory(name)
        with patch.dict('os.environ', {'CRITEO_RESULTS_MODE': 'typo'}):
            with self.assertRaises(ValueError):
                result_directory('run')
        with TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                result_directory('absent', d)

    def test_synthetic_demo_real_learners_freeze_selection_and_no_overwrite(self):
        with TemporaryDirectory() as d:
            out = Path(d) / 'demo'
            policies = run_demo(out, n=1600, trees=3, bootstrap=5)
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertFalse(manifest['is_criteo_result'])
            self.assertEqual(manifest['dataset'], 'synthetic randomized demonstration')
            self.assertTrue(manifest['selection']['frozen_before_test'])
            self.assertEqual(len(policies), 6)
            self.assertTrue(np.isfinite(policies.incremental_per_100k).all())
            self.assertEqual(int(policies.n_selected.iloc[0]), 64)
            with self.assertRaises(FileExistsError):
                run_demo(out, n=1600)
