import copy
import hashlib
import json
import unittest

import numpy as np

from sim.factory.replicator_plan import sample_plan
from src.audit_replicator_dataset import compare_seeded_replay


class ReplicatorPlanTests(unittest.TestCase):
    def test_seed_reproduces_exact_parameter_sequence(self):
        first = np.random.default_rng(2026)
        second = np.random.default_rng(2026)
        plans_a = [sample_plan(first, index, 2.73).as_dict() for index in range(12)]
        plans_b = [sample_plan(second, index, 2.73).as_dict() for index in range(12)]
        self.assertEqual(plans_a, plans_b)

    def test_plan_stays_inside_safe_canonical_bounds(self):
        generator = np.random.default_rng(7)
        for index in range(50):
            plan = sample_plan(generator, index, 2.73)
            self.assertTrue(0.480 <= plan.object_position[0] <= 0.520)
            self.assertTrue(-0.025 <= plan.object_position[1] <= 0.025)
            self.assertTrue(1.24 <= plan.camera_position[2] <= 1.36)
            self.assertTrue(0.50 <= plan.object_size_fraction <= 0.76)
            self.assertEqual(len(plan.clutter_positions), 3)
            self.assertTrue(0.18 <= plan.plate_roughness <= 0.72)

    def test_replay_comparison_fails_on_parameter_drift(self):
        plan = sample_plan(np.random.default_rng(11), 0, 2.73).as_dict()
        encoded = json.dumps(plan, sort_keys=True, separators=(",", ":"))
        plan_hash = hashlib.sha256(encoded.encode()).hexdigest()
        row = {
            "group": "source-a",
            "content_hash": "crop-a",
            "extra": json.dumps({
                "view": 0,
                "full_frame_sha256": "full-a",
                "replicator_plan": plan,
                "replicator_plan_sha256": plan_hash,
            }),
        }
        replay = copy.deepcopy(row)
        result = compare_seeded_replay([row], [replay])
        self.assertEqual(result["parameter_matches"], 1)

        changed = copy.deepcopy(plan)
        changed["key_intensity"] += 1.0
        replay["extra"] = json.dumps({
            "view": 0,
            "full_frame_sha256": "full-a",
            "replicator_plan": changed,
            "replicator_plan_sha256": hashlib.sha256(
                json.dumps(changed, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        })
        with self.assertRaisesRegex(ValueError, "parameter replay mismatch"):
            compare_seeded_replay([row], [replay])


if __name__ == "__main__":
    unittest.main()
