import torch


class RigidBody:
    def __init__(self, batch_size: int, device: str = 'cpu'):
        self.batch_size = batch_size
        self.device = device
        self._state = torch.zeros((batch_size, 4), device=device)

    @property
    def position(self) -> torch.Tensor:
        return self._state[:, :2]

    @position.setter
    def position(self, value: torch.Tensor) -> None:
        self._state[:, :2] = value

    @property
    def velocity(self) -> torch.Tensor:
        return self._state[:, 2]

    @velocity.setter
    def velocity(self, value: torch.Tensor) -> None:
        self._state[:, 2] = value

    @property
    def rotation(self) -> torch.Tensor:
        return self._state[:, 3]

    @rotation.setter
    def rotation(self, value: torch.Tensor) -> None:
        self._state[:, 3] = value

    def update(self, dt: float) -> None:
        pass


class Puck(RigidBody):
    def __init__(self, batch_size: int = 32, device: str = 'cpu'):
        super().__init__(batch_size, device)

    def update(self, dt: float) -> None:
        self._state[:, 0] += self.velocity * self.rotation.cos() * dt
        self._state[:, 1] += self.velocity * self.rotation.sin() * dt


class Kart(RigidBody):
    def __init__(self, batch_size: int = 32, device: str = 'cpu'):
        super().__init__(batch_size, device)
        self._command = torch.zeros((batch_size, 3), device=device)
        self.max_speed = 25.0
        self.min_speed = -0.5 * self.max_speed
        self.time_to_max_speed = 2.0
        self.max_acceleration = self.max_speed / self.time_to_max_speed
        self.brake_acceleration = -self.max_acceleration
        self.drag_coefficient = self.max_acceleration / self.max_speed / self.max_speed
        self.min_turn_radius = 2.0
        self.turn_radius_slope = 28.0 / 45.0
        self.wheel_base = 1.2

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @command.setter
    def command(self, value: torch.Tensor) -> None:
        self._command = value

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

    def update(self, dt: float) -> None:
        k1 = self.derivative(self._state, self._command)
        k2 = self.derivative(self._state + 0.5 * dt * k1, self._command)
        k3 = self.derivative(self._state + 0.5 * dt * k2, self._command)
        k4 = self.derivative(self._state + dt * k3, self._command)
        self._state += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self._state[:, 2] = self.limit_speed(self._state[:, 2])
