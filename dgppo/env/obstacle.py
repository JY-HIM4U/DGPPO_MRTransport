import jax
import jax.numpy as jnp

from typing import NamedTuple, Protocol
from jax.scipy.spatial.transform import Rotation
from dgppo.utils.typing import Pos2d, Pos3d, Pos
from dgppo.utils.typing import (Array, ObsType, ObsWidth, ObsHeight, ObsTheta,
                                Radius, ObsLength, ObsQuaternion, BoolScalar)

RECTANGLE = jnp.zeros(1)
CUBOID = jnp.ones(1)
SPHERE = jnp.ones(1) * 2
CIRCLE = jnp.array(3)  # Define a new obstacle type
POLYGON = jnp.ones(1) * 4


class Obstacle(Protocol):
    type: ObsType
    center: Pos

    @property
    def n(self) -> int:
        pass

    def inside(self, point: Pos, r: Radius = 0.) -> BoolScalar:
        pass

    def raytracing(self, start: Pos, end: Pos) -> Array:
        pass


class Rectangle(NamedTuple):
    type: ObsType
    center: Pos2d
    width: ObsWidth
    height: ObsHeight
    theta: ObsTheta
    points: Array

    @staticmethod
    def create(center: Pos2d, width: ObsWidth, height: ObsHeight, theta: ObsTheta) -> "Rectangle":
        bbox = jnp.array([
            [width / 2, height / 2],
            [-width / 2, height / 2],
            [-width / 2, -height / 2],
            [width / 2, -height / 2],
        ]).T  # (2, 4)

        rot = jnp.array([
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)]
        ])

        trans = center[:, None]
        points = jnp.dot(rot, bbox) + trans
        points = points.T

        return Rectangle(RECTANGLE, center, width, height, theta, points)

    @property
    def n(self) -> int:
        return self.center.shape[0]

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        rel_x = point[0] - self.center[0]
        rel_y = point[1] - self.center[1]
        rel_xx = jnp.abs(rel_x * jnp.cos(self.theta) + rel_y * jnp.sin(self.theta)) - self.width / 2
        rel_yy = jnp.abs(rel_x * jnp.sin(self.theta) - rel_y * jnp.cos(self.theta)) - self.height / 2
        is_in_down = jnp.logical_and(rel_xx < r, rel_yy < 0)
        is_in_up = jnp.logical_and(rel_xx < 0, rel_yy < r)
        is_out_corner = jnp.logical_and(rel_xx > 0, rel_yy > 0)
        is_in_circle = jnp.sqrt(rel_xx ** 2 + rel_yy ** 2) < r
        is_in = jnp.logical_or(jnp.logical_or(is_in_down, is_in_up), jnp.logical_and(is_out_corner, is_in_circle))
        return is_in

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        # beam
        x1 = start[0]
        y1 = start[1]
        x2 = end[0]
        y2 = end[1]

        # edges
        x3 = self.points[:, 0]
        y3 = self.points[:, 1]
        x4 = self.points[[-1, 0, 1, 2], 0]
        y4 = self.points[[-1, 0, 1, 2], 1]

        '''
        # solve the equation
        # x = x1 + alpha * (x2 - x1) = x3 + beta * (x4 - x3)
        # y = y1 + alpha * (y2 - y1) = y3 + beta * (y4 - y3)
        # equivalent to solve
        # [x1-x2 x4-x3] alpha  = [x1-x3]
        # [y1-y2 y4-y4] beta     [y1-y3]
        # solve by (alpha beta)^T = A^{-1} b
        '''

        det = (x1 - x2) * (y4 - y3) - (y1 - y2) * (x4 - x3)
        # clip det for numerical issues
        det = jnp.sign(det) * jnp.clip(jnp.abs(det), 1e-7, 1e7)
        alphas = ((y4 - y3) * (x1 - x3) - (x4 - x3) * (y1 - y3)) / det
        betas = (-(y1 - y2) * (x1 - x3) + (x1 - x2) * (y1 - y3)) / det
        valids = jnp.logical_and(jnp.logical_and(alphas <= 1, alphas >= 0), jnp.logical_and(betas <= 1, betas >= 0))
        alphas = valids * alphas + (1 - valids) * 1e6
        alphas = jnp.min(alphas)  # reduce the polygon edges dimension
        return alphas


class Polygon(NamedTuple):
    """Arbitrary 2-D simple polygon with a fixed, padded vertex count.

    ``points`` holds ``max_verts`` vertices; a polygon with fewer real vertices
    pads by repeating its *last* real vertex.  Edges are taken as
    ``(points[i], points[i - 1])`` (same convention as ``Rectangle``), so the
    padding produces the correct closing edge plus zero-length edges, which are
    masked out everywhere below.
    """
    type: ObsType
    center: Pos2d
    points: Array  # (max_verts, 2)

    @staticmethod
    def create(points: Array, center: Pos2d = None) -> "Polygon":
        points = jnp.asarray(points)
        if center is None:
            center = points.mean(axis=0)
        return Polygon(POLYGON, center, points)

    @property
    def n(self) -> int:
        return self.center.shape[0]

    def _edges(self) -> tuple[Array, Array, Array, Array, Array]:
        """Returns x3, y3, x4, y4 of every edge plus a non-degenerate mask."""
        x3 = self.points[:, 0]
        y3 = self.points[:, 1]
        prev = jnp.roll(self.points, 1, axis=0)
        x4 = prev[:, 0]
        y4 = prev[:, 1]
        ok = (jnp.abs(x4 - x3) + jnp.abs(y4 - y3)) > 1e-9
        return x3, y3, x4, y4, ok

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        x, y = point[0], point[1]
        x3, y3, x4, y4, ok = self._edges()

        # even-odd crossing test against the horizontal ray y = const, x -> +inf
        straddles = jnp.logical_and((y3 > y) != (y4 > y), ok)
        dy = y4 - y3
        dy_safe = jnp.where(jnp.abs(dy) < 1e-12, 1e-12, dy)
        x_cross = x3 + (y - y3) * (x4 - x3) / dy_safe
        crossings = jnp.logical_and(straddles, x < x_cross)
        is_in = (jnp.sum(crossings) % 2) == 1

        # inflate by r: also count points within r of the boundary
        ex, ey = x4 - x3, y4 - y3
        l2 = jnp.maximum(ex ** 2 + ey ** 2, 1e-12)
        t = jnp.clip(((x - x3) * ex + (y - y3) * ey) / l2, 0., 1.)
        dist = jnp.sqrt((x - (x3 + t * ex)) ** 2 + (y - (y3 + t * ey)) ** 2)
        dist = jnp.where(ok, dist, jnp.inf)
        near_boundary = jnp.min(dist) <= r

        return jnp.logical_or(is_in, near_boundary)

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        x1, y1 = start[0], start[1]
        x2, y2 = end[0], end[1]
        x3, y3, x4, y4, ok = self._edges()

        # same linear system as Rectangle.raytracing, but degenerate (padded)
        # edges give det == 0, which the sign() trick there would turn into a
        # divide-by-zero, so they are clipped and masked explicitly.
        det = (x1 - x2) * (y4 - y3) - (y1 - y2) * (x4 - x3)
        det = jnp.where(jnp.abs(det) < 1e-7, 1e-7, det)
        alphas = ((y4 - y3) * (x1 - x3) - (x4 - x3) * (y1 - y3)) / det
        betas = (-(y1 - y2) * (x1 - x3) + (x1 - x2) * (y1 - y3)) / det
        valids = jnp.logical_and(
            jnp.logical_and(alphas <= 1, alphas >= 0),
            jnp.logical_and(betas <= 1, betas >= 0))
        valids = jnp.logical_and(valids, ok)
        alphas = jnp.where(valids, alphas, 1e6)
        return jnp.min(alphas)


class Cuboid(NamedTuple):
    type: ObsType
    center: Pos3d
    length: ObsLength
    width: ObsWidth
    height: ObsHeight
    rotation: Rotation
    points: Array

    @staticmethod
    def create(
            center: Pos3d, length: ObsLength, width: ObsWidth, height: ObsHeight, quaternion: ObsQuaternion
    ) -> "Cuboid":
        bbox = jnp.array([
            [-length / 2, -width / 2, -height / 2],
            [length / 2, -width / 2, -height / 2],
            [length / 2, width / 2, -height / 2],
            [-length / 2, width / 2, -height / 2],
            [-length / 2, -width / 2, height / 2],
            [length / 2, -width / 2, height / 2],
            [length / 2, width / 2, height / 2],
            [-length / 2, width / 2, height / 2],
        ])  # (8, 3)

        rotation = Rotation.from_quat(quaternion)
        points = rotation.apply(bbox) + center
        return Cuboid(CUBOID, center, length, width, height, rotation, points)

    def n(self) -> int:
        return self.center.shape[0]

    def inside(self, point: Pos3d, r: Radius = 0.) -> BoolScalar:
        # transform the point to the cuboid frame
        rot = self.rotation.as_matrix()
        rot_inv = jnp.linalg.inv(rot)
        point = jnp.dot(rot_inv, point - self.center)

        # check if the point is inside the cuboid
        is_in_height = ((-self.length / 2 < point[0]) & (point[0] < self.length / 2)) & \
                       ((-self.width / 2 < point[1]) & (point[1] < self.width / 2)) & \
                       ((-self.height / 2 - r < point[2]) & (point[2] < self.height / 2 + r))
        is_in_length = ((-self.length / 2 - r < point[0]) & (point[0] < self.length / 2 + r)) & \
                       ((-self.width / 2 < point[1]) & (point[1] < self.width / 2)) & \
                       ((-self.height / 2 < point[2]) & (point[2] < self.height / 2))
        is_in_width = ((-self.length / 2 < point[0]) & (point[0] < self.length / 2)) & \
                      ((-self.width / 2 - r < point[1]) & (point[1] < self.width / 2 + r)) & \
                      ((-self.height / 2 < point[2]) & (point[2] < self.height / 2))
        is_in = is_in_height | is_in_length | is_in_width

        # check if the sphere intersects with the edges
        edge_order = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0],
                                [4, 5], [5, 6], [6, 7], [7, 4],
                                [0, 4], [1, 5], [2, 6], [3, 7]])
        edges = self.points[edge_order]

        def intersect_edge(edge: Array) -> BoolScalar:
            assert edge.shape == (2, 3)
            dot_prod = jnp.dot(edge[1] - edge[0], point - edge[0])
            frac = dot_prod / ((jnp.linalg.norm(edge[1] - edge[0])) ** 2)
            frac = jnp.clip(frac, 0, 1)
            closest_point = edge[0] + frac * (edge[1] - edge[0])
            dist = jnp.linalg.norm(closest_point - point)
            return dist <= r

        is_intersect = jnp.any(jax.vmap(intersect_edge)(edges))
        return is_in | is_intersect

    def raytracing(self, start: Pos3d, end: Pos3d) -> Array:
        # beams
        x1, y1, z1 = start[0], start[1], start[2]
        x2, y2, z2 = end[0], end[1], end[2]

        # those are for edges
        # point order for the base 0~7 (first 0-x-xy-y for the lower level, then 0-x-xy-y for the upper level)
        # face order: bottom, left, right, upper, outer left, outer right
        x3 = self.points[[0, 0, 0, 6, 6, 6], 0]
        y3 = self.points[[0, 0, 0, 6, 6, 6], 1]
        z3 = self.points[[0, 0, 0, 6, 6, 6], 2]

        x4 = self.points[[1, 1, 3, 5, 5, 7], 0]
        y4 = self.points[[1, 1, 3, 5, 5, 7], 1]
        z4 = self.points[[1, 1, 3, 5, 5, 7], 2]

        x5 = self.points[[3, 4, 4, 7, 2, 2], 0]
        y5 = self.points[[3, 4, 4, 7, 2, 2], 1]
        z5 = self.points[[3, 4, 4, 7, 2, 2], 2]

        '''
        # solve the equation
        # x = x1 + alpha * (x2 - x1) = x3 + beta * (x4 - x3) + gamma * (x5 - x3)
        # y = y1 + alpha * (y2 - y1) = y3 + beta * (y4 - y3) + gamma * (y5 - y3) 
        # z = z1 + alpha * (z2 - z1) = z3 + beta * (z4 - z3) + gamma * (z5 - z3)
        # equivalent to solve
        # [x1 - x2  x4 - x3  x5 - x3]  alpha  =  [x1 - x3] 
        # [y1 - y2  y4 - y3  y5 - y3]  beta      [y1 - y3]
        # [z1 - z2  z4 - z3  z5 - z3]  gamma     [z1 - z3]
        # solve by (alpha beta gamma)^T = A^{-1} b

        # A^{-1} = 1/det * [(y4-y3)*(z5-z3)-(y5-y3)*(z4-z3)      -[(x4-x3)*(z5-z3)-(z4-z3)*(x5-x3)]      (x4-x3)*(y5-y3)-(y4-y3)*(x5-x3)]
        #                  [-[(y1-y2)*(z5-z3)-(z1-z2)*(y5-y3)]   (x1-x2)*(z5-z3)-(z1-z2)*(x5-x3)      -[(x1-x2)*(y5-y3)-(y1-y2)*(x5-x3)]]
        #                  [(y1-y2)*(z4-z3)-(y4-y3)*(z1-z2)      -[(x1-x2)*(z4-z3)-(z1-z2)*(x4-x3)]      (x1-x2)*(y4-y3)-(y1-y2)*(x4-x3)]
        '''

        det = (x1 - x2) * (y4 - y3) * (z5 - z3) + (x4 - x3) * (y5 - y3) * (z1 - z2) + (y1 - y2) * (z4 - z3) * (
                x5 - x3) - (y1 - y2) * (x4 - x3) * (z5 - z3) - (z4 - z3) * (y5 - y3) * (x1 - x2) - (x5 - x3) * (
                      y4 - y3) * (z1 - z2)
        # clip det for numerical issues
        det = jnp.sign(det) * jnp.clip(jnp.abs(det), 1e-7, 1e7)
        adj_00 = (y4 - y3) * (z5 - z3) - (y5 - y3) * (z4 - z3)
        adj_01 = -((x4 - x3) * (z5 - z3) - (z4 - z3) * (x5 - x3))
        adj_02 = (x4 - x3) * (y5 - y3) - (y4 - y3) * (x5 - x3)
        adj_10 = -((y1 - y2) * (z5 - z3) - (z1 - z2) * (y5 - y3))
        adj_11 = (x1 - x2) * (z5 - z3) - (z1 - z2) * (x5 - x3)
        adj_12 = -((x1 - x2) * (y5 - y3) - (y1 - y2) * (x5 - x3))
        adj_20 = (y1 - y2) * (z4 - z3) - (y4 - y3) * (z1 - z2)
        adj_21 = -((x1 - x2) * (z4 - z3) - (z1 - z2) * (x4 - x3))
        adj_22 = (x1 - x2) * (y4 - y3) - (y1 - y2) * (x4 - x3)
        alphas = 1 / det * (adj_00 * (x1 - x3) + adj_01 * (y1 - y3) + adj_02 * (z1 - z3))
        betas = 1 / det * (adj_10 * (x1 - x3) + adj_11 * (y1 - y3) + adj_12 * (z1 - z3))
        gammas = 1 / det * (adj_20 * (x1 - x3) + adj_21 * (y1 - y3) + adj_22 * (z1 - z3))
        valids = jnp.logical_and(
            jnp.logical_and(jnp.logical_and(alphas <= 1, alphas >= 0), jnp.logical_and(betas <= 1, betas >= 0)),
            jnp.logical_and(gammas <= 1, gammas >= 0)
        )
        alphas = valids * alphas + (1 - valids) * 1e6
        alphas = jnp.min(alphas)  # reduce the polygon edges dimension
        return alphas


class Sphere(NamedTuple):
    type: ObsType
    center: Pos3d
    radius: Radius

    @staticmethod
    def create(center: Pos3d, radius: Radius) -> "Sphere":
        return Sphere(SPHERE, center, radius)

    def n(self) -> int:
        return self.center.shape[0]

    def inside(self, point: Pos3d, r: Radius = 0.) -> BoolScalar:
        return jnp.linalg.norm(point - self.center) <= self.radius + r

    def raytracing(self, start: Pos3d, end: Pos3d) -> Array:
        x1, y1, z1 = start[0], start[1], start[2]
        x2, y2, z2 = end[0], end[1], end[2]
        xc, yc, zc = self.center[0], self.center[1], self.center[2]
        r = self.radius

        '''
        # solve the equation
        # x = x1 + alpha * (x2 - x1) = xc + r * sin(gamma) * cos(theta)
        # y = y1 + alpha * (y2 - y1) = yc + r * sin(gamma) * sin(theta)
        # z = z1 + alpha * (z2 - z1) = zc + r * cos(gamma)
        # equivalent to solve (eliminate theta using sin^2(sin^2+cos^2) +cos^2 ...=1)
        # [(x2-x1)^2+(y2-y1)^2+(z2-z1)^2]alpha^2+2[(x2-x1)(x1-xc)+(y2-y1)(y1-yc)+(z2-z1)(z1-zc)]alpha+(x1-xc)^2+(y1-yc)^2+(z1-zc)^2-r^2=0
        # A alpha^2 + B alpha + C = 0
        # check delta = B^2-4AC
        # alpha = ...
        # take valid min
        '''
        lidar_rmax = jnp.linalg.norm(end - start)
        A = lidar_rmax ** 2  # (x2-x1)^2+(y2-y1)^2
        B = 2 * ((x2 - x1) * (x1 - xc) + (y2 - y1) * (y1 - yc) + (z2 - z1) * (z1 - zc))
        C = (x1 - xc) ** 2 + (y1 - yc) ** 2 + (z1 - zc) ** 2 - r ** 2

        delta = B ** 2 - 4 * A * C
        valid1 = delta >= 0

        alpha1 = (-B - jnp.sqrt(delta * valid1)) / (2 * A) * valid1 + (1 - valid1)
        alpha2 = (-B + jnp.sqrt(delta * valid1)) / (2 * A) * valid1 + (1 - valid1)
        alpha1_tilde = (alpha1 >= 0) * alpha1 + (alpha1 < 0) * 1
        alpha2_tilde = (alpha2 >= 0) * alpha2 + (alpha2 < 0) * 1
        alphas = jnp.minimum(alpha1_tilde, alpha2_tilde)
        alphas = jnp.clip(alphas, 0, 1)
        alphas = valid1 * alphas + (1 - valid1) * 1e6
        return alphas


class Circle(NamedTuple):
    type: ObsType
    center: Pos2d
    radius: Radius

    @staticmethod
    def create(center: Pos2d, radius: Radius) -> "Circle":
        return Circle(CIRCLE, center, radius)

    @property
    def n(self) -> int:
        return self.center.shape[0]

    def inside(self, point: Pos2d, r: Radius = 0.) -> BoolScalar:
        return jnp.linalg.norm(point - self.center) <= self.radius + r

    def raytracing(self, start: Pos2d, end: Pos2d) -> Array:
        x1, y1 = start[0], start[1]
        x2, y2 = end[0], end[1]
        xc, yc = self.center[0], self.center[1]
        r = self.radius

        '''
        # solve the equation
        # x = x1 + alpha * (x2 - x1) = xc + r * cos(theta)
        # y = y1 + alpha * (y2 - y1) = yc + r * sin(theta)
        # equivalent to solve (eliminate theta using sin^2+cos^2=1)
        # [(x2-x1)^2+(y2-y1)^2]alpha^2+2[(x2-x1)(x1-xc)+(y2-y1)(y1-yc)]alpha+(x1-xc)^2+(y1-yc)^2-r^2=0
        # A alpha^2 + B alpha + C = 0
        # check delta = B^2-4AC
        # alpha = ...
        # take valid min
        '''
        lidar_rmax = jnp.linalg.norm(end - start)
        A = lidar_rmax ** 2  # (x2-x1)^2+(y2-y1)^2
        B = 2 * ((x2 - x1) * (x1 - xc) + (y2 - y1) * (y1 - yc))
        C = (x1 - xc) ** 2 + (y1 - yc) ** 2 - r ** 2

        delta = B ** 2 - 4 * A * C
        valid1 = delta >= 0

        alpha1 = (-B - jnp.sqrt(delta * valid1)) / (2 * A) * valid1 + (1 - valid1)
        alpha2 = (-B + jnp.sqrt(delta * valid1)) / (2 * A) * valid1 + (1 - valid1)
        alpha1_tilde = (alpha1 >= 0) * alpha1 + (alpha1 < 0) * 1
        alpha2_tilde = (alpha2 >= 0) * alpha2 + (alpha2 < 0) * 1
        alphas = jnp.minimum(alpha1_tilde, alpha2_tilde)
        alphas = jnp.clip(alphas, 0, 1)
        alphas = valid1 * alphas + (1 - valid1) * 1e6
        return alphas
