import re
from typing import Set, Tuple, Dict

from causal_discovery.graph.node import Node


class BackgroundKnowledge(object):
    def __init__(self):
        self.forbidden_rules_specs: Set[Tuple[Node, Node]] = set()
        self.forbidden_pattern_rules_specs: Set[Tuple[str, str]] = set()
        self.required_rules_specs: Set[Tuple[Node, Node]] = set()
        self.required_pattern_rules_specs: Set[Tuple[str, str]] = set()
        self.tier_map: Dict[int, Set[Node]] = {}
        self.tier_value_map: Dict[Node, int] = {}

    def add_forbidden_by_node(self, node1: Node, node2: Node):
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))
        self.forbidden_rules_specs.add((node1, node2))
        return self

    def add_required_by_node(self, node1: Node, node2: Node):
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))
        self.required_rules_specs.add((node1, node2))
        return self

    def add_forbidden_by_pattern(self, node_pattern1: str, node_pattern2: str):
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str. node_pattern1 = ' + str(
                type(node_pattern1)) + ' node_pattern2 = ' + str(type(node_pattern2)))
        self.forbidden_pattern_rules_specs.add((node_pattern1, node_pattern2))
        return self

    def add_required_by_pattern(self, node_pattern1: str, node_pattern2: str):
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str. node_pattern1 = ' + str(
                type(node_pattern1)) + ' node_pattern2 = ' + str(type(node_pattern2)))
        self.required_pattern_rules_specs.add((node_pattern1, node_pattern2))
        return self

    def _ensure_tiers(self, tier: int):
        if type(tier) != int:
            raise TypeError('tier must be int type. tier = ' + str(type(tier)))
        for t in range(tier + 1):
            if t not in self.tier_map:
                self.tier_map[t] = set()

    def add_node_to_tier(self, node: Node, tier: int):
        if (not isinstance(node, Node)) or type(tier) != int:
            raise TypeError(
                'node must be instance of Node. tier must be int type. node = ' + str(type(node)) + ' tier = ' + str(
                    type(tier)))
        if tier < 0:
            raise TypeError('tier must be a non-negative integer. tier = ' + str(tier))
        self._ensure_tiers(tier)
        self.tier_map.get(tier).add(node)
        self.tier_value_map[node] = tier
        return self

    def _is_node_match_regular_expression(self, pattern: str, node: Node) -> bool:
        return re.match(pattern, node.get_name()) is not None

    def is_forbidden(self, node1: Node, node2: Node) -> bool:
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError('node1 and node2 must be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(
                type(node2)))

        # first check in forbidden_rules_specs
        for (from_node, to_node) in self.forbidden_rules_specs:
            if from_node == node1 and to_node == node2:
                return True

        # then check in forbidden_pattern_rules_specs
        for (from_node_pattern, to_node_pattern) in self.forbidden_pattern_rules_specs:
            if self._is_node_match_regular_expression(from_node_pattern,
                                                      node1) and self._is_node_match_regular_expression(to_node_pattern,
                                                                                                        node2):
                return True

        # then check in tier_map -- FIXED: use > not >= so same-tier edges are allowed
        if node1 in self.tier_value_map and node2 in self.tier_value_map:
            if self.tier_value_map[node1] > self.tier_value_map[node2]:
                return True

        return False

    def is_required(self, node1: Node, node2: Node) -> bool:
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError('node1 and node2 must be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(
                type(node2)))

        for (from_node, to_node) in self.required_rules_specs:
            if from_node == node1 and to_node == node2:
                return True

        for (from_node_pattern, to_node_pattern) in self.required_pattern_rules_specs:
            if self._is_node_match_regular_expression(from_node_pattern,
                                                      node1) and self._is_node_match_regular_expression(to_node_pattern,
                                                                                                        node2):
                return True

        return False

    def remove_forbidden_by_node(self, node1: Node, node2: Node):
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))
        if (node1, node2) in self.forbidden_rules_specs:
            self.forbidden_rules_specs.remove((node1, node2))
        return self

    def remove_required_by_node(self, node1: Node, node2: Node):
        if (not isinstance(node1, Node)) or (not isinstance(node2, Node)):
            raise TypeError(
                'node must not be instance of Node. node1 = ' + str(type(node1)) + ' node2 = ' + str(type(node2)))
        if (node1, node2) in self.required_rules_specs:
            self.required_rules_specs.remove((node1, node2))
        return self

    def remove_forbidden_by_pattern(self, node_pattern1: str, node_pattern2: str):
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str.')
        if (node_pattern1, node_pattern2) in self.forbidden_pattern_rules_specs:
            self.forbidden_pattern_rules_specs.remove((node_pattern1, node_pattern2))
        return self

    def remove_required_by_pattern(self, node_pattern1: str, node_pattern2: str):
        if type(node_pattern1) != str or type(node_pattern2) != str:
            raise TypeError('node_pattern must be type of str.')
        if (node_pattern1, node_pattern2) in self.required_pattern_rules_specs:
            self.required_pattern_rules_specs.remove((node_pattern1, node_pattern2))
        return self

    def remove_node_from_tier(self, node: Node, tier: int):
        if (not isinstance(node, Node)) or type(tier) != int:
            raise TypeError(
                'node must be instance of Node. tier must be int type.')
        if tier < 0:
            raise TypeError('tier must be a non-negative integer. tier = ' + str(tier))
        self._ensure_tiers(tier)
        if node in self.tier_map.get(tier):
            self.tier_map.get(tier).remove(node)
        if node in self.tier_value_map:
            self.tier_value_map.pop(node)
        return self

    def is_in_which_tier(self, node: Node) -> int:
        return self.tier_value_map[node] if node in self.tier_value_map else -1
