import unittest
import torch

from .physics import Puck, Kart


class TorchAssertions:
    def assertTorchAlmostEqual(
            self,
            first: torch.Tensor,
            second: torch.Tensor,
            delta: float = 1e-7) -> None:
        # diff = (first - second).abs()
        # check = diff >= delta
        # found = check.any()
        if ((first - second).abs() >= delta).any():
            raise AssertionError('expected {}\nfound {}'.format(first, second))


class TestPuck(unittest.TestCase, TorchAssertions):
    def test_position(self):
        position = torch.zeros((5, 2))
        velocity = torch.tensor((1.0, 2.0, 3.0, 4.0, 5.0))
        rotation = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0)) * torch.pi
        puck = Puck(position, velocity, rotation)

        command = torch.zeros((5, 3))
        dt = 1.0
        puck.update(dt)

        # check direction is correct
        self.assertTorchAlmostEqual(
            torch.stack((rotation.cos(), rotation.sin()), dim=1),
            puck.position() / puck.position().norm(dim=1).unsqueeze(1)
        )

        # check magnitude is correct
        self.assertTorchAlmostEqual(
            velocity * dt,
            puck.position().norm(dim=1),
            1.0e-6
        )


class TestKart(unittest.TestCase, TorchAssertions):
    def test_position(self):
        position = torch.zeros((5, 2))
        velocity = torch.tensor((1.0, 2.0, 3.0, 4.0, 5.0))
        rotation = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0)) * torch.pi
        kart = Kart(position, velocity, rotation)

        command = torch.zeros((5, 3))
        dt = 1.0
        kart.update(command, dt)

        # check direction is correct
        self.assertTorchAlmostEqual(
            torch.stack((rotation.cos(), rotation.sin()), dim=1),
            kart.position() / kart.position().norm(dim=1).unsqueeze(1)
        )

        # check magnitude is correct
        self.assertTorchAlmostEqual(
            velocity * dt,
            kart.position().norm(dim=1),
            1.0e-6
        )

    def test_no_acceleration(self):
        position = torch.zeros((5, 2))
        velocity = torch.tensor((-1.0, 0.0, 1.0, 2.0, 3.0))
        rotation = torch.zeros(5)
        kart = Kart(position, velocity, rotation)

        command = torch.zeros((5, 3))
        dt = 1.0
        kart.update(command, dt)

        self.assertTorchAlmostEqual(velocity, kart.velocity())

    def test_acceleration(self):
        position = torch.zeros((5, 2))
        velocity = torch.zeros(5)
        rotation = torch.zeros(5)
        kart = Kart(position, velocity, rotation)

        command = torch.tensor((
            (0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (1.0, 0.0, 0.0)
        ))
        dt = kart.time_to_max_speed
        kart.update(command, dt)

        # test speed
        self.assertTorchAlmostEqual(command[:, 0] * kart.max_speed, kart.velocity())

        # test max
        kart.update(command, dt)
        self.assertTrue((kart.velocity() <= kart.max_speed).all())

    def test_brake(self):
        position = torch.zeros((5, 2))
        velocity = torch.zeros(5)
        rotation = torch.zeros(5)
        kart = Kart(position, velocity, rotation)

        command = torch.tensor((0.0, 1.0, 0.0)).repeat((5, 1))
        dt = 0.5 * kart.time_to_max_speed
        kart.update(command, dt)

        # test speed
        self.assertTorchAlmostEqual(kart.min_speed, kart.velocity())

        # test max
        kart.update(command, dt)
        self.assertTrue((kart.velocity() >= kart.min_speed).all())

    def test_steer(self):
        position = torch.zeros((3, 2))
        velocity = torch.tensor((0.0, 5.0, 25.0))
        rotation = torch.zeros(3)
        kart = Kart(position, velocity, rotation)

        command = torch.tensor((
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ))
        dt = 1.0
        kart.update(command, dt)

        self.assertTorchAlmostEqual(velocity, kart.velocity())
        self.assertTorchAlmostEqual(rotation, kart.rotation())

        command = torch.tensor((
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ))
        kart.update(command, dt)

        rot = kart.rotation()
        self.assertAlmostEqual(kart.rotation()[0], 0.0)
        self.assertTrue(kart.rotation()[1] > 0.0)
        self.assertTrue(kart.rotation()[2] > 0.0)
        self.assertTrue(kart.rotation()[1] / velocity[1] > kart.rotation()[2] / velocity[2])

        command = torch.tensor((
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
        ))
        dt = 2.0
        kart.update(command, dt)

        self.assertAlmostEqual(kart.rotation()[0], 0.0)
        self.assertTrue(kart.rotation()[1] < 0.0)
        self.assertTrue(kart.rotation()[2] < 0.0)
        self.assertTrue(kart.rotation()[1] / velocity[1] < kart.rotation()[2] / velocity[2])

    def test_back_up(self):
        position = torch.zeros((2, 2))
        velocity = torch.tensor((-5.0, -12.5))
        rotation = torch.zeros(2)
        kart = Kart(position, velocity, rotation)

        command = torch.tensor((
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ))
        dt = 1.0
        kart.update(command, dt)

        self.assertTrue(kart.rotation()[0] < 0.0)
        self.assertTrue(kart.rotation()[1] < 0.0)

        command = torch.tensor((
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
        ))
        dt = 2.0
        kart.update(command, dt)

        self.assertTrue(kart.rotation()[0] > 0.0)
        self.assertTrue(kart.rotation()[1] > 0.0)


if __name__ == '__main__':
    unittest.main()
