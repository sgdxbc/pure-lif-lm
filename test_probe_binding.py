"""Known-answer and traffic checks for the CPU binding probe."""
import unittest

from experiments.probe_binding import CAP, Coincidence, DigitalTable, OuterProduct, Traffic


class BindingTest(unittest.TestCase):
    def test_reverse_key_and_dropped_event(self):
        a, b = (1, 2, 3, 4), (31, 32, 33, 34)
        for builder in (Coincidence, lambda: DigitalTable(True), lambda: OuterProduct(2048)):
            model = builder()
            model.write(a, b, 7, 1, Traffic())
            model.write(b, a, 21, 2, Traffic())
            self.assertEqual(model.query(a, b, 2050, 0, Traffic()), 7)
            self.assertEqual(model.query(b, a, 2050, 0, Traffic()), 21)
            self.assertEqual(model.query(a[1:], b, 2050, 0, Traffic()), 7)
            self.assertLessEqual(model.state_bytes, CAP)

    def test_hand_count_single_contact(self):
        model = Coincidence()
        write, read = Traffic(), Traffic()
        model.write((1,), (2,), 11, 0, write)
        self.assertEqual((write.read, write.written), (4, 4))
        self.assertEqual(model.query((1,), (2,), 3, 0, read), 11)
        self.assertEqual((read.read, read.written), (291, 257))
        self.assertEqual(read.bytes(), 548)

    def test_outer_product_without_decay_writes_only_the_active_value(self):
        model = OuterProduct(float("inf"))
        write, read = Traffic(), Traffic()
        model.write((1,), (2,), 11, 0, write)
        self.assertEqual((write.read, write.written), (4, 4))
        self.assertEqual(model.query((1,), (2,), 3, 0, read), 11)
        self.assertEqual((read.read, read.written), (384, 256))

    def test_no_coincidence_creates_no_synapse(self):
        model = Coincidence()
        model.write((), (2,), 11, 0, Traffic())
        self.assertTrue((model.values == 0).all())


if __name__ == "__main__":
    unittest.main()
