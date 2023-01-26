import csv
import datetime
import math
import sys
from collections import Counter
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Sequence

import click
import matplotlib.pyplot as plt  # type: ignore
import networkx as nx  # type: ignore

from samplers.mcmc_sampler import MCMCSampler as Sampler

# from samplers.random_sampler import RandomSampler as Sampler
from samplers.sampler import Node

TRANSFER_STEP_PER_SAMPLING = 100
SAMPLING_PER_EXECUTION = 10000
LOGGING_INTERVAL_BETWEEN_SAMPLING = 1000

SCORE_COEFFICIENT = 1e5

VISUALIZER_SEED = 2434
VISUALIZER_INTENSITY = 0.25
NODE_SIZE = 800
EDGE_COLOR = "0.7"

logger = getLogger(__name__)
basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=INFO)


def load_graph(graph_path: Path) -> nx.Graph:
    logger.info("start loading graph at %s", graph_path)

    graph = nx.Graph(name=graph_path.name)

    with open(graph_path) as f:
        N, M = f.readline().split()

        for i in range(int(N)):
            weight = f.readline()
            graph.add_node(i, weight=int(weight))  # linear weight
            # graph.add_node(i, weight=math.exp(int(weight)))  # exp weight

        for i in range(int(M)):
            u, v = f.readline().split()
            graph.add_edge(int(u), int(v))

    logger.info(
        "graph is loaded; %d nodes and %d edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )

    return graph


def sample_and_count_repeatedly(graph: nx.Graph) -> Counter[int]:
    def build_node_list(graph: nx.Graph, node_indexes: Sequence[int]) -> list[Node]:
        return [
            Node(index=index, degree=graph.degree(index), weight=weight)
            for index, weight in graph.nodes(data="weight")
            if index in node_indexes
        ]

    sampler = Sampler()
    sampled_node_counter: Counter[int] = Counter()
    context = -1

    for num_sampling in range(SAMPLING_PER_EXECUTION):
        if num_sampling % LOGGING_INTERVAL_BETWEEN_SAMPLING == 0:
            logger.info("complete sampling %d times", num_sampling)

        current_node, context = sampler.initialize(
            nodes=build_node_list(graph=graph, node_indexes=graph.nodes()),
            context=context,
        )

        for _ in range(TRANSFER_STEP_PER_SAMPLING):
            adjacent_nodes = build_node_list(
                graph=graph, node_indexes=list(graph.neighbors(current_node.index))
            )
            current_node, context = sampler.transfer(
                current_node=current_node,
                adjacent_nodes=adjacent_nodes,
                context=context,
                transfer_step_per_sampling=TRANSFER_STEP_PER_SAMPLING,
            )

        sampled_node_counter[current_node.index] += 1

    logger.info(
        "complete sampling %d times (whole sampling is done)", SAMPLING_PER_EXECUTION
    )

    return sampled_node_counter


def compute_expected_sampling_count(graph: nx.Graph, v: int, num_sampling: int) -> int:
    sum_weight = sum(dict(graph.nodes(data="weight")).values())
    return int(graph.nodes(data="weight")[v] * num_sampling / sum_weight)


def compute_score(
    graph: nx.Graph, sampled_node_counter: Counter[int], num_sampling: int
) -> int:
    assert num_sampling > 0

    kl_divergence: float = 0.0
    for v in graph.nodes():
        expected_ratio = (
            compute_expected_sampling_count(graph=graph, v=v, num_sampling=num_sampling)
            / num_sampling
        )
        actual_ratio = sampled_node_counter[v] / num_sampling
        kl_divergence += actual_ratio * (
            math.log2(actual_ratio + 1e-9) - math.log2(expected_ratio + 1e-9)
        )

    return -int(math.log2(kl_divergence) * SCORE_COEFFICIENT)


def save_image(
    image_path: Path, graph: nx.Graph, sampled_node_counter: Counter[int]
) -> None:
    def decide_node_color(v: int) -> float:
        expected_count = max(
            compute_expected_sampling_count(
                graph=graph, v=v, num_sampling=SAMPLING_PER_EXECUTION
            ),
            1,
        )
        actual_count = max(sampled_node_counter[v], 1)
        log_ratio = math.log2(actual_count / expected_count)
        return max(0.0, min(1.0, VISUALIZER_INTENSITY * log_ratio + 0.5))

    pos = nx.spring_layout(graph, seed=VISUALIZER_SEED)
    nx.draw_networkx(
        graph,
        pos,
        node_color=[decide_node_color(v) for v in graph.nodes()],
        edge_color=EDGE_COLOR,
        node_size=NODE_SIZE,
        cmap=plt.cm.coolwarm,
        vmin=0.0,
        vmax=1.0,
        labels=dict(graph.nodes(data="weight")),
    )
    font = {"fontname": "Helvetica", "color": "k", "fontweight": "bold", "fontsize": 14}
    ax = plt.gca()
    ax.set_title(f"graph={graph.name}, sampler={Sampler.__name__}", font)
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(image_path)


def save_count_csv(
    csv_path: Path, graph: nx.Graph, sampled_node_counter: Counter[int]
) -> None:
    with open(csv_path, mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "expected", "actual", "ratio"])
        for v in graph.nodes():
            expected_count = max(
                compute_expected_sampling_count(
                    graph=graph, v=v, num_sampling=SAMPLING_PER_EXECUTION
                ),
                1,
            )
            actual_count = max(sampled_node_counter[v], 1)
            ratio = actual_count / expected_count
            writer.writerow([v, expected_count, actual_count, ratio])


def save_output(
    output_dir_path: Path, graph: nx.Graph, sampled_node_counter: Counter[int]
) -> None:
    output_dir_path.mkdir(exist_ok=True)

    save_image(
        image_path=output_dir_path / "graph.png",
        graph=graph,
        sampled_node_counter=sampled_node_counter,
    )
    save_count_csv(
        csv_path=output_dir_path / "count.csv",
        graph=graph,
        sampled_node_counter=sampled_node_counter,
    )


@click.command
@click.option("--graph-path", type=Path, required=True)
@click.option(
    "--output-dir-path",
    type=Path,
    default=Path("outputs")
    / str(datetime.datetime.now()).split(".")[0].replace(" ", "-").replace(":", "-"),
)
def vertex_sampler(graph_path: Path, output_dir_path: Path) -> None:
    graph = load_graph(graph_path)

    if not nx.is_connected(graph):
        logger.warning("only connected graph is acceptable")
        sys.exit(0)

    logger.info("use %s as sampler", Sampler.__name__)
    logger.info(
        "TRANSFER_STEP_PER_SAMPLING=%d, SAMPLING_PER_EXECUTION=%d",
        TRANSFER_STEP_PER_SAMPLING,
        SAMPLING_PER_EXECUTION,
    )

    sampled_node_counter = sample_and_count_repeatedly(graph)

    logger.info(
        "sampling score is %d",
        compute_score(
            graph=graph,
            sampled_node_counter=sampled_node_counter,
            num_sampling=SAMPLING_PER_EXECUTION,
        ),
    )

    save_output(
        output_dir_path=output_dir_path,
        graph=graph,
        sampled_node_counter=sampled_node_counter,
    )


if __name__ == "__main__":
    vertex_sampler()
