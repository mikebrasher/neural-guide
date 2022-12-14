import unittest
import torch

from .physics import RigidBody


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


class TestApply(unittest.TestCase, TorchAssertions):
    def test_position(self):
        position = torch.zeros((5, 2))
        velocity = torch.tensor((1.0, 2.0, 3.0, 4.0, 5.0))
        rotation = torch.tensor((0.0, 0.25, 0.5, 0.75, 1.0)) * torch.pi
        rb = RigidBody(position, velocity, rotation)

        command = torch.zeros((5, 3))
        dt = 1.0
        rb.update(command, dt)

        # check direction is correct
        self.assertTorchAlmostEqual(
            torch.stack((rotation.cos(), rotation.sin()), dim=1),
            rb.position() / rb.position().norm(dim=1).unsqueeze(1)
        )

        # check magnitude is correct
        self.assertTorchAlmostEqual(
            velocity * dt,
            rb.position().norm(dim=1),
            1.0e-6
        )

    def test_no_acceleration(self):
        position = torch.zeros((5, 2))
        velocity = torch.tensor((-1.0, 0.0, 1.0, 2.0, 3.0))
        rotation = torch.zeros(5)
        rb = RigidBody(position, velocity, rotation)

        command = torch.zeros((5, 3))
        dt = 1.0
        rb.update(command, dt)

        self.assertTorchAlmostEqual(velocity, rb.velocity())

    def test_acceleration(self):
        position = torch.zeros((5, 2))
        velocity = torch.zeros(5)
        rotation = torch.zeros(5)
        rb = RigidBody(position, velocity, rotation)

        command = torch.tensor((
            (0.0, 0.0, 0.0),
            (0.25, 0.0, 0.0),
            (0.5, 0.0, 0.0),
            (0.75, 0.0, 0.0),
            (1.0, 0.0, 0.0)
        ))
        dt = rb.time_to_max_speed
        rb.update(command, dt)

        # test speed
        self.assertTorchAlmostEqual(command[:, 0] * rb.max_speed, rb.velocity())

        # test max
        rb.update(command, dt)
        self.assertTrue((rb.velocity() <= rb.max_speed).all())

    def test_brake(self):
        position = torch.zeros((5, 2))
        velocity = torch.zeros(5)
        rotation = torch.zeros(5)
        rb = RigidBody(position, velocity, rotation)

        command = torch.tensor((0.0, 1.0, 0.0)).repeat((5, 1))
        dt = 0.5 * rb.time_to_max_speed
        rb.update(command, dt)

        # test speed
        self.assertTorchAlmostEqual(rb.min_speed, rb.velocity())

        # test max
        rb.update(command, dt)
        self.assertTrue((rb.velocity() >= rb.min_speed).all())

    def test_steer(self):
        position = torch.zeros((3, 2))
        velocity = torch.tensor((0.0, 5.0, 25.0))
        rotation = torch.zeros(3)
        rb = RigidBody(position, velocity, rotation)

        command = torch.tensor((
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ))
        dt = 1.0
        rb.update(command, dt)

        self.assertTorchAlmostEqual(velocity, rb.velocity())
        self.assertTorchAlmostEqual(rotation, rb.rotation())

        command = torch.tensor((
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ))
        rb.update(command, dt)

        rot = rb.rotation()
        self.assertAlmostEqual(rb.rotation()[0], 0.0)
        self.assertTrue(rb.rotation()[1] > 0.0)
        self.assertTrue(rb.rotation()[2] > 0.0)
        self.assertTrue(rb.rotation()[1] / velocity[1] > rb.rotation()[2] / velocity[2])

        command = torch.tensor((
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
        ))
        dt = 2.0
        rb.update(command, dt)

        self.assertAlmostEqual(rb.rotation()[0], 0.0)
        self.assertTrue(rb.rotation()[1] < 0.0)
        self.assertTrue(rb.rotation()[2] < 0.0)
        self.assertTrue(rb.rotation()[1] / velocity[1] < rb.rotation()[2] / velocity[2])

    def test_back_up(self):
        position = torch.zeros((2, 2))
        velocity = torch.tensor((-5.0, -12.5))
        rotation = torch.zeros(2)
        rb = RigidBody(position, velocity, rotation)

        command = torch.tensor((
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        ))
        dt = 1.0
        rb.update(command, dt)

        self.assertTrue(rb.rotation()[0] < 0.0)
        self.assertTrue(rb.rotation()[1] < 0.0)

        command = torch.tensor((
            (0.0, 0.0, -1.0),
            (0.0, 0.0, -1.0),
        ))
        dt = 2.0
        rb.update(command, dt)

        self.assertTrue(rb.rotation()[0] > 0.0)
        self.assertTrue(rb.rotation()[1] > 0.0)


if __name__ == '__main__':
    unittest.main()
