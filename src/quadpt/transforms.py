import functools
import itertools
import math
import numpy as np
import operator
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abstractmethod
    def backward(self, y):
        raise NotImplementedError

    @abstractmethod
    def jac(self, x):
        raise NotImplementedError

    @abstractmethod
    def jacinv(self, y):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)

    def __matmul__(self, other):
        if not isinstance(other, Transform):
            raise TypeError("Can only compose with another Transform.")
        return CompositeTransform(self, other)

    @classmethod
    def chain(cls, *transforms):
        """Chain multiple transforms together."""
        return CompositeTransform(*transforms)


class Affine(Transform):
    def __init__(self, scale=1.0, shift=0.0):
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.scale * x + self.shift

    def backward(self, y):
        return (y - self.shift) / self.scale

    def jac(self, x):
        return self.scale

    def jacinv(self, y):
        return 1.0 / self.scale

    def __matmul__(self, other):
        if isinstance(other, Affine):
            # higher precision
            return Affine(scale=self.scale * other.scale, shift=self.scale * other.shift + self.shift)
        return super().__matmul__(other)

    @classmethod
    def regularize(cls, a, b):
        """Map [a, b] to [-1, 1]."""
        return cls(2.0 / (b - a), (a + b) / (a - b))

    @classmethod
    def map_to(cls, prev, new):
        """Map prev to new"""
        (a, b), (c, d) = prev, new
        return cls(scale=(c - d) / (a - b), shift=(a * d - b * c) / (a - b))


class Logarithm(Transform):
    def __init__(self, base=math.e):
        if base not in [math.e, 10, 2]:
            raise ValueError("Base must be e, 10, or 2.")
        self.base = base
        self.log = {math.e: np.log, 10: np.log10, 2: np.log2}[base]

    def forward(self, x):
        return self.log(x)

    def backward(self, y):
        return self.base**y

    def jac(self, x):
        return 1 / (x * np.log(self.base))

    def jacinv(self, y):
        return self.base**y * np.log(self.base)


class CompositeTransform(Transform):
    def __init__(self, *transforms):
        transforms = list(transforms)
        transforms = itertools.chain.from_iterable(
            transf.transforms if isinstance(transf, CompositeTransform) else [transf] for transf in transforms
        )
        # combine Affines
        reduced_transforms = []
        for isaffine, group in itertools.groupby(transforms, key=lambda t: isinstance(t, Affine)):
            if isaffine:
                combined = functools.reduce(operator.matmul, group)
                reduced_transforms.append(combined)
            else:
                reduced_transforms.extend(group)
        self.transforms = reduced_transforms

    def forward(self, x):
        for transf in reversed(self.transforms):
            x = transf.forward(x)
        return x

    def backward(self, y):
        for transf in self.transforms:
            y = transf.backward(y)
        return y

    def jac(self, x):
        jacobian = 1.0
        for transf in reversed(self.transforms):
            jacobian *= transf.jac(x)
            x = transf.forward(x)
        return jacobian

    def jacinv(self, y):
        jacobian_inv = 1.0
        for transf in self.transforms:
            jacobian_inv *= transf.jacinv(y)
            y = transf.backward(y)
        return jacobian_inv
