#!/usr/bin/env python3
from enum import Enum


class NodeType(Enum):
    MEASURED = 1
    LATENT = 2
    ERROR = 3
    SESSION = 4
    RANDOMIZE = 5
    LOCK = 6
    NO_TYPE = 7

    def __str__(self):
        return self.name


class Node:
    """Abstract base class for graph nodes."""

    def get_name(self) -> str:
        pass

    def set_name(self, name: str):
        pass

    def get_node_type(self) -> NodeType:
        pass

    def set_node_type(self, node_type: NodeType):
        pass

    def __str__(self):
        pass

    def get_center_x(self) -> int:
        pass

    def set_center_x(self, center_x: int):
        pass

    def get_center_y(self) -> int:
        pass

    def set_center_y(self, center_y: int):
        pass

    def set_center(self, center_x: int, center_y: int):
        pass

    def __hash__(self):
        pass

    def __eq__(self, other):
        pass

    def like(self, name: str):
        pass

    def get_all_attributes(self):
        pass

    def get_attribute(self, key):
        pass

    def remove_attribute(self, key):
        pass

    def add_attribute(self, key, value):
        pass
