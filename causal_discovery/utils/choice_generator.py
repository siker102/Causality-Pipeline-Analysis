from __future__ import annotations

from typing import List


class ChoiceGenerator:
    """Generates all combinations of a choose b."""

    def __init__(self, a: int, b: int):
        self.a = a
        self.b = b
        self.diff = a - b
        self.choiceLocal: List[int] = []
        for i in range(b - 1):
            self.choiceLocal.append(i)
        if b > 0:
            self.choiceLocal.append(b - 2)
        self.choiceReturned: List[int] = [0 for i in range(b)]
        self.begun = False

    def fill(self, index: int):
        self.choiceLocal[index] += 1
        for i in range(index + 1, self.b):
            self.choiceLocal[i] = self.choiceLocal[i - 1] + 1

    def next(self) -> List[int] | None:
        i = self.b

        while i > 0:
            i -= 1
            if self.choiceLocal[i] < (i + self.diff):
                self.fill(i)
                self.begun = True
                for j in range(self.b):
                    self.choiceReturned[j] = self.choiceLocal[j]
                return self.choiceReturned

        if self.begun:
            return None
        else:
            self.begun = True
            for j in range(self.b):
                self.choiceReturned[j] = self.choiceLocal[j]
            return self.choiceReturned


class DepthChoiceGenerator:
    """Generates all combinations up to a given depth."""

    def _initialize(self):
        self.diff = self.a - self.b
        self.choiceLocal: List[int] = []

        for i in range(self.b - 1):
            self.choiceLocal.append(i)
        if self.b > 0:
            self.choiceLocal.append(self.b - 2)

        self.choiceReturned: List[int] = [0 for i in range(self.b)]
        self.begun = False

    def __init__(self, a: int, depth: int):
        if a < 0 or depth < -1:
            raise Exception("Illegal Argument!")

        self.a = a
        self.b = 0
        self.depth = depth

        self.effectiveDepth = depth
        if depth == -1:
            self.effectiveDepth = a
        if depth > a:
            self.effectiveDepth = a

        self._initialize()

    def _fill(self, index: int):
        self.choiceLocal[index] += 1
        for i in range(index + 1, self.b):
            self.choiceLocal[i] = self.choiceLocal[i - 1] + 1

    def next(self) -> List[int] | None:
        i = self.b

        while i > 0:
            i -= 1
            if self.choiceLocal[i] < i + self.diff:
                self._fill(i)
                self.begun = True
                for j in range(self.b):
                    self.choiceReturned[j] = self.choiceLocal[j]
                return self.choiceReturned

        if self.begun:
            self.b += 1

            if self.b > self.effectiveDepth:
                return None

            self._initialize()
            return self.next()
        else:
            self.begun = True
            for j in range(self.b):
                self.choiceReturned[j] = self.choiceLocal[j]
            return self.choiceReturned
