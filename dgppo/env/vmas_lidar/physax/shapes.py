import jax.numpy as jnp
import jax

from .jax_types import Pos2
import jax.lax as lax

def safe_assert(condition, message: str):
    # Define a branch that does nothing if condition is True…
    def ok(_):
        return None
    # …and a branch that raises an error if condition is False.
    def not_ok(_):
        # Raise an error outside of JIT when the branch is executed.
        raise ValueError(message)
    # Use lax.cond to select between the branches.
    # Note: When running under JIT, the branch not taken will not be executed.
    lax.cond(condition, ok, not_ok, operand=None)


class Shape:
    def moment_of_inertia(self, mass: float):
        raise NotImplementedError

# class Object(Shape):
#     def __init__(self, length: float = 0.3):
#         super().__init__()
#            assert length > 0, f"Length must be > 0, got {length}"
#         self._length = length  # length from center to vertex
#         self.hollow = False    # triangles are always hollow in this case
from jax import core

class Object(Shape):
    def __init__(self, length: float = 0.3):
        super().__init__()
        # Only perform the check if length is a concrete value.
        if not isinstance(length, core.Tracer):
            assert length > 0, f"Length must be > 0, got {length}"
        self._length = length  # length from center to vertex
        self.hollow = False    # triangles are always hollow in this case

    @property
    def length(self):
        return self._length

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        # For triangle, anchor points are at the vertices
        # Each vertex is length distance from center at angles 0, 2π/3, 4π/3
        angle = anchor[1] * (2 * jnp.pi / 3)  # maps anchor[1] to angle
        return self.length * jnp.array([jnp.cos(angle), jnp.sin(angle)])

    def moment_of_inertia(self, mass: float):
        # For an equilateral triangle about its center
        # I = (1/12) * mass * length^2
        return (1 / 12) * mass * (self.length**2)

    def circumscribed_radius(self):
        # For an equilateral triangle, circumradius equals length
        # (distance from center to vertex)
        return self.length

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #     
    #     # Calculate vertices of equilateral triangle
    #     angles = jnp.array([0, 2*jnp.pi/3, 4*jnp.pi/3])
    #     vertices = [(self.length * jnp.cos(angle), 
    #                 self.length * jnp.sin(angle)) 
    #                for angle in angles]
    #     return rendering.make_polygon(vertices)


class Box(Shape):
    def __init__(self, length: float = 0.3, width: float = 0.1, hollow: bool = False):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        assert width > 0, f"Width must be > 0, got {length}"
        self._length = length
        self._width = width
        self.hollow = hollow

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        return anchor * jnp.array([self.length / 2, self.width / 2])
        # return anchor[X] * self.length / 2, anchor[Y] * self.width / 2

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2 + self.width**2)

    def circumscribed_radius(self):
        return jnp.sqrt((self.length / 2) ** 2 + (self.width / 2) ** 2)

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #
    #     l, r, t, b = (
    #         -self.length / 2,
    #         self.length / 2,
    #         self.width / 2,
    #         -self.width / 2,
    #     )
    #     return rendering.make_polygon([(l, b), (l, t), (r, t), (r, b)])


class Sphere(Shape):
    def __init__(self, radius: float = 0.05):
        super().__init__()
        assert radius > 0, f"Radius must be > 0, got {radius}"
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        delta = self.radius * anchor
        delta_norm = jnp.linalg.vector_norm(delta)
        # If delta_norm is larger than the radius, scale it down to 1 / radius.
        delta_mult = jnp.where(delta_norm > self.radius, 1 / (self.radius * delta_norm), 1)
        delta = delta * delta_mult
        # if delta_norm > self.radius:
        #     delta /= delta_norm * self.radius
        return delta

    def moment_of_inertia(self, mass: float):
        return (1 / 2) * mass * self.radius**2

    def circumscribed_radius(self):
        return self.radius

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #
    #     return rendering.make_circle(self.radius)


class Line(Shape):
    def __init__(self, length: float = 0.5):
        super().__init__()
        assert length > 0, f"Length must be > 0, got {length}"
        self._length = length
        self._width = 2

    @property
    def length(self):
        return self._length

    @property
    def width(self):
        return self._width

    def moment_of_inertia(self, mass: float):
        return (1 / 12) * mass * (self.length**2)

    def circumscribed_radius(self):
        return self.length / 2

    def get_delta_from_anchor(self, anchor: Pos2) -> Pos2:
        return anchor[0] * self.length / 2, 0.0

    # def get_geometry(self) -> "Geom":
    #     from vmas.simulator import rendering
    #
    #     return rendering.Line(
    #         (-self.length / 2, 0),
    #         (self.length / 2, 0),
    #         width=self.width,
    #     )
