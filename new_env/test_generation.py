import random
import statistics
import unittest
import prepare_dataset as prep

class SequenceRNG:
    def __init__(self, values): self.values = iter(values)
    def gauss(self, mean, sd): return next(self.values)

class GenerationTests(unittest.TestCase):
    def test_rejects_both_tails_instead_of_clipping(self):
        self.assertEqual(prep.sample_soc(SequenceRNG([.1, .5, .31])), .31)
    def test_bounds_are_inclusive(self):
        self.assertEqual(prep.sample_soc(SequenceRNG([.2])), .2)
        self.assertEqual(prep.sample_soc(SequenceRNG([.4])), .4)
    def test_invalid_parameters_fail(self):
        for options in ({'sd': 0}, {'mean': .1}, {'mean': float('nan')}, {'low': .5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                prep.sample_soc(random.Random(42), **options)
    def test_fixed_seed_reproduces_exact_soc(self):
        seed = prep.power_seed(20261107, 2650, 'p01', 50000, 'EV_7')
        self.assertEqual(prep.sample_soc(random.Random(seed)), prep.sample_soc(random.Random(seed)))
    def test_identity_fields_have_independent_streams(self):
        original = (20261107, 2650, 'p01', 50000, 'EV_7')
        options = [original, (20261108, *original[1:]), (original[0], 2800, *original[2:]),
                   (*original[:2], 'p02', *original[3:]), (*original[:3], 50001, original[4]), (*original[:4], 'EV_9')]
        self.assertEqual(len({prep.power_seed(*key) for key in options}), len(options))
    def test_large_sample_has_expected_truncated_shape(self):
        rng = random.Random(42)
        values = [prep.sample_soc(rng) for _ in range(50000)]
        self.assertTrue(all(.2 <= value <= .4 for value in values))
        self.assertAlmostEqual(statistics.mean(values), .3, delta=.0008)
        self.assertAlmostEqual(statistics.stdev(values), .04398, delta=.0008)
        self.assertGreater(sum(.28 <= v <= .32 for v in values), sum(v <= .22 or v >= .38 for v in values))

if __name__ == '__main__': unittest.main(verbosity=2)
