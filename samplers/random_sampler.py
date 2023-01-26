import random
from typing import Sequence

from samplers.sampler import Node, Sampler


class RandomSampler(Sampler):
    def initialize(self, nodes: Sequence[Node], context: int) -> tuple[Node, int]:
        return nodes[0], 0

    def transfer(
        self,
        current_node: Node,
        adjacent_nodes: Sequence[Node],
        context: int,
        transfer_step_per_sampling: int,
    ) -> tuple[Node, int]:
        """
        0.5 の確率で移動せず、0.5 の確率で隣接点をランダムに選んで移動する
        """
        return (
            random.choice([current_node] * len(adjacent_nodes) + list(adjacent_nodes)),
            context * 1,
        )
