from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Node:
    index: int
    degree: int
    weight: int


class Sampler(metaclass=ABCMeta):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize(self, nodes: Sequence[Node], context: int) -> tuple[Node, int]:
        raise NotImplementedError()

    @abstractmethod
    def transfer(
        self,
        current_node: Node,
        adjacent_nodes: Sequence[Node],
        context: int,
        transfer_step_per_sampling: int,
    ) -> tuple[Node, int]:
        raise NotImplementedError()
