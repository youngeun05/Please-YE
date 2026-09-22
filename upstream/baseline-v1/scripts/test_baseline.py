"""CPU contract tests: fixed splits, leakage rejection and evaluator compatibility."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
import types
from baseline_common import ROOT, check_splits, check_names, kitti_row
from evaluate_kitti import parse_label
from train_baseline import validate_data


class BaselineTests(unittest.TestCase):
    def test_repository_splits(self):
        self.assertEqual(tuple(map(len, check_splits())), (5481, 1000, 1000))

    def test_names_reject_coco_and_reordering(self):
        check_names({0: 'Car', 1: 'Pedestrian', 2: 'Cyclist'})
        for names in [['person', 'car', 'bicycle'], ['Cyclist', 'Pedestrian', 'Car']]:
            with self.assertRaises(ValueError):
                check_names(names)

    def test_detection_contract(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / '000000.txt'
            rows = ''.join(kitti_row(i, [1.25, 2, 30, 45], 0.81) for i in range(3))
            self.assertTrue(all(len(row.split()) == 16 for row in rows.splitlines()))
            path.write_text(rows, encoding='utf-8')
            parsed = parse_label(path, True)
            self.assertEqual(parsed['name'].tolist(), ['Car', 'Pedestrian', 'Cyclist'])
            self.assertAlmostEqual(parsed['score'][0], 0.81)
            self.assertEqual(parsed['bbox'][0].tolist(), [1.25, 2, 30, 45])
            path.write_text('', encoding='utf-8')
            self.assertEqual(parse_label(path, True)['bbox'].shape, (0, 4))

    def test_actual_manifest_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / 'splits').mkdir()
            for name, value in [('train', '000001'), ('internal_val', '000002'), ('calibration', '000001')]:
                (root / 'splits' / f'{name}.txt').write_text(value)
            (root / 'eval_val.txt').write_text('000003')
            base = root / 'data'
            for sub in ['images/all', 'labels/all', 'lists']:
                (base / sub).mkdir(parents=True)
            for i in ['000001', '000002', '000003']:
                (base / 'images/all' / f'{i}.png').touch()
                (base / 'labels/all' / f'{i}.txt').touch()
            (base / 'conversion_report.local.json').write_text(json.dumps({'neighbor_policy': 'ignore'}))
            config = {'path': str(base), 'names': ['Car', 'Pedestrian', 'Cyclist'],
                      'train': 'lists/train.txt', 'val': 'lists/val.txt'}
            config_path = base / 'test.yaml'
            config_path.write_text(json.dumps(config))
            train = base / 'lists/train.txt'
            train.write_text(str(base / 'images/all/000001.png'))
            (base / 'lists/val.txt').write_text(str(base / 'images/all/000002.png'))
            # JSON is a YAML subset; mock only the YAML dependency on this minimal host.
            with patch.dict(sys.modules, {'yaml': types.SimpleNamespace(safe_load=json.loads)}):
                self.assertEqual(validate_data(config_path, root)['train'], 1)
                train.write_text(str(base / 'images/all/000003.png'))
                with self.assertRaisesRegex(ValueError, 'Leakage guard'):
                    validate_data(config_path, root)
                train.write_text('\n'.join([str(base / 'images/all/000001.png')] * 2))
                with self.assertRaisesRegex(ValueError, 'Leakage guard'):
                    validate_data(config_path, root)


if __name__ == '__main__':
    unittest.main()
