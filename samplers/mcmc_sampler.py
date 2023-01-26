import random
from typing import Sequence

from samplers.sampler import Node, Sampler


class MCMCSampler(Sampler):
    def initialize(self, nodes: Sequence[Node], context: int) -> tuple[Node, int]:
        return nodes[0], 0

    def transfer(
        self,
        current_node: Node,
        adjacent_nodes: Sequence[Node],
        context: int,
        transfer_step_per_sampling: int,
    ) -> tuple[Node, int]:
        candidate_node = random.choice(adjacent_nodes)

        transfer_prob = (
            current_node.degree
            / current_node.weight
            * min(
                current_node.weight / current_node.degree,
                candidate_node.weight / candidate_node.degree,
            )
        )

        return (
            (candidate_node, context + 1)
            if random.random() < transfer_prob
            else (current_node, context + 1)
        )

    # def transfer(
    #     self,
    #     current_node: Node,
    #     adjacent_nodes: Sequence[Node],
    #     context: int,
    #     transfer_step_per_sampling: int,
    # ) -> tuple[Node, int]:
    #     """
    #     序盤はランダムに動き回ることで局所解を抜け出す作戦をとる
    #     """
    #     if context < transfer_step_per_sampling * 0.42:
    #         candidate_node = random.choice(adjacent_nodes)
    #         return candidate_node, context + 1
    #     else:
    #         candidate_node = random.choice(adjacent_nodes)

    #         transfer_prob = (
    #             current_node.degree
    #             / current_node.weight
    #             * min(
    #                 current_node.weight / current_node.degree,
    #                 candidate_node.weight / candidate_node.degree,
    #             )
    #         )

    #         return (
    #             (candidate_node, context + 1)
    #             if random.random() < transfer_prob
    #             else (current_node, context + 1)
    #         )
