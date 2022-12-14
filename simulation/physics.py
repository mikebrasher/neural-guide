import torch


class RigidBody:
    def __init__(
            self,
            position: torch.Tensor,  # batch_size x 2
            velocity: torch.Tensor,  # batch_size x 1
            rotation: torch.Tensor,  # batch_size x 1
    ):
        self.max_speed = 25.0
        self.min_speed = -0.5 * self.max_speed
        self.time_to_max_speed = 2.0
        self.max_acceleration = self.max_speed / self.time_to_max_speed
        self.brake_acceleration = -self.max_acceleration
        self.drag_coefficient = self.max_acceleration / self.max_speed / self.max_speed
        self.min_turn_radius = 2.0
        self.turn_radius_slope = 28.0 / 45.0
        self.wheel_base = 1.2
        self.state = torch.cat(
            (
                position,
                self.limit_speed(velocity).unsqueeze(1),
                rotation.unsqueeze(1)
            ), dim=1)

    def position(self) -> torch.Tensor:
        return self.state[:, :2]

    def velocity(self) -> torch.Tensor:
        return self.state[:, 2]

    def rotation(self) -> torch.Tensor:
        return self.state[:, 3]

    def limit_speed(self, velocity: torch.Tensor) -> torch.Tensor:
        return torch.clip(velocity, self.min_speed, self.max_speed)

    def derivative(self, state: torch.Tensor, command: torch.tensor) -> torch.Tensor:
        velocity = self.limit_speed(state[:, 2])
        sigma = state[:, 3]

        accel_command = torch.clip(command[:, 0], 0, 1)
        brake_command = command[:, 1]
        accel = accel_command * self.max_acceleration
        accel[brake_command == 1.0] = self.brake_acceleration

        steer_command = torch.clip(command[:, 2], -1, 1)
        min_turn_radius_at_speed = self.min_turn_radius + velocity.abs() * self.turn_radius_slope
        max_steer_angle = (self.wheel_base / min_turn_radius_at_speed).asin()
        delta = steer_command * max_steer_angle

        x_dot = velocity * sigma.cos()
        y_dot = velocity * sigma.sin()
        velocity_dot = accel
        sigma_dot = delta.sin() * velocity / self.wheel_base
        return torch.stack((x_dot, y_dot, velocity_dot, sigma_dot), dim=1)

    def update(self, command: torch.Tensor, dt: float) -> None:
        k1 = self.derivative(self.state, command)
        k2 = self.derivative(self.state + 0.5 * dt * k1, command)
        k3 = self.derivative(self.state + 0.5 * dt * k2, command)
        k4 = self.derivative(self.state + dt * k3, command)
        self.state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.state[:, 2] = self.limit_speed(self.state[:, 2])
