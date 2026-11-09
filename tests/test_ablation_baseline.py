"""The final modified run, rather than mutable repository defaults, anchors new plans."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from ablation_study import load_modified_baseline
from experiment_config import load_experiment, backend_settings, config_hash


class ModifiedBaselineTests(unittest.TestCase):
    def save(self, root):
        experiment = copy.deepcopy(load_experiment())
        experiment['common'].update(batch_size=30, gradient_accumulation_steps=1, learning_rate=0.00015)
        saved = dict(backend_settings(experiment, 'cuda'), mode='reason',
                     experiment=experiment, experiment_sha256=config_hash(experiment))
        (root/'training_config.json').write_text(json.dumps(saved))
        (root/'completed.json').write_text(json.dumps({'smoke': False, 'epoch': 3}))
        return saved

    def test_saved_recipe_is_authoritative_and_copied(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);saved=self.save(root)
            baseline=load_modified_baseline(root)
            self.assertEqual(baseline['common']['batch_size'],30)
            self.assertEqual(baseline['common']['learning_rate'],0.00015)
            baseline['common']['batch_size']=12
            self.assertEqual(json.loads((root/'training_config.json').read_text()),saved)

    def test_reject_inconsistent_effective_settings_or_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for key,value in [('batch_size',12),('experiment_sha256','changed')]:
                saved=self.save(root);saved[key]=value
                (root/'training_config.json').write_text(json.dumps(saved))
                with self.assertRaises(ValueError):load_modified_baseline(root)

    def test_reject_smoke_or_wrong_model_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);self.save(root)
            (root/'completed.json').write_text(json.dumps({'smoke':True}))
            with self.assertRaises(ValueError):load_modified_baseline(root)
            saved=self.save(root);saved['mode']='simple'
            (root/'training_config.json').write_text(json.dumps(saved))
            with self.assertRaises(ValueError):load_modified_baseline(root)


if __name__=='__main__':unittest.main()
