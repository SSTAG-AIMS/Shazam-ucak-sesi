import unittest

import numpy as np

from aircraft_agent_pilot_v1 import feature_vector


class AircraftAgentPilotTests(unittest.TestCase):
    def test_feature_vector_is_finite_and_fixed_size(self):
        vector = feature_vector(np.zeros(22050, dtype=np.float32))
        self.assertEqual(vector.shape, (130,))
        self.assertTrue(np.all(np.isfinite(vector)))


if __name__ == "__main__":
    unittest.main()
