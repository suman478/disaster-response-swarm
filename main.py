"""
Single-file multi-agent swarm simulation for disaster response.

Run with:
    python main.py
"""

import random
import time
import math
from typing import List, Set, Tuple, Optional

Coord = Tuple[int, int]

# Cell types
EMPTY = 0
OBSTACLE = 1
VICTIM = 2


# =========================
# World (2D grid)
# =========================
class World:
    """Simple 2D grid world with obstacles and victims."""

    def __init__(
        self,
        width: int,
        height: int,
        obstacle_density: float,
        num_victims: int,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.rng = random.Random(seed)

        # grid[y][x]
        self.grid: List[List[int]] = [
            [EMPTY for _ in range(width)] for _ in range(height)
        ]

        self.obstacles: Set[Coord] = set()
        self.victims: Set[Coord] = set()

        self._generate_obstacles(obstacle_density)
        self._place_victims(num_victims)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, x: int, y: int) -> bool:
        return (x, y) in self.obstacles

    def is_free(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and not self.is_obstacle(x, y)

    def _generate_obstacles(self, density: float) -> None:
        num_cells = self.width * self.height
        num_obstacles = int(num_cells * max(0.0, min(density, 0.8)))
        placed = 0
        while placed < num_obstacles:
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            if (x, y) not in self.obstacles:
                self.obstacles.add((x, y))
                self.grid[y][x] = OBSTACLE
                placed += 1

    def _place_victims(self, num_victims: int) -> None:
        placed = 0
        attempts = 0
        max_attempts = num_victims * 50

        while placed < num_victims and attempts < max_attempts:
            attempts += 1
            x = self.rng.randrange(self.width)
            y = self.rng.randrange(self.height)
            if (x, y) in self.obstacles or (x, y) in self.victims:
                continue
            self.victims.add((x, y))
            self.grid[y][x] = VICTIM
            placed += 1

    def get_neighbors_4(self, x: int, y: int) -> List[Coord]:
        """4-connected neighbors (up, down, left, right) within bounds."""
        candidates = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        return [(nx, ny) for nx, ny in candidates if self.in_bounds(nx, ny)]

    def render_ascii(self, agent_positions: Optional[Set[Coord]] = None) -> str:
        """
        Simple ASCII representation:
        '.' empty
        '#' obstacle
        'V' victim
        'A' agent
        """
        agent_positions = agent_positions or set()
        lines: List[str] = []

        for y in range(self.height):
            row_chars: List[str] = []
            for x in range(self.width):
                pos = (x, y)
                if pos in agent_positions:
                    row_chars.append("A")
                else:
                    cell = self.grid[y][x]
                    if cell == OBSTACLE:
                        row_chars.append("#")
                    elif cell == VICTIM:
                        row_chars.append("V")
                    else:
                        row_chars.append(".")
            lines.append("".join(row_chars))
        return "\n".join(lines)


# =========================
# Agent
# =========================
class Agent:
    """Simple agent with local knowledge and basic movement."""

    def __init__(
        self,
        agent_id: int,
        position: Coord,
        sensing_range: int,
        comm_range: int,
    ) -> None:
        self.id = agent_id
        self.position: Coord = position
        self.sensing_range = sensing_range
        self.comm_range = comm_range

        # Knowledge
        self.visited: Set[Coord] = {position}
        self.known_victims: Set[Coord] = set()

    def update_position(self, new_pos: Coord) -> None:
        self.position = new_pos
        self.visited.add(new_pos)

    def integrate_victim_info(self, victim_positions: Set[Coord]) -> None:
        self.known_victims |= victim_positions


# =========================
# Communication
# =========================
def euclidean_distance(a: Agent, b: Agent) -> float:
    ax, ay = a.position
    bx, by = b.position
    return math.hypot(ax - bx, ay - by)


def get_neighbors(agent: Agent, agents: List[Agent]) -> List[Agent]:
    """Agents within communication range of the given agent."""
    return [
        other
        for other in agents
        if other is not agent and euclidean_distance(agent, other) <= agent.comm_range
    ]


def share_victim_information(agents: List[Agent]) -> None:
    """
    Simple gossip: each agent merges victim info
    from neighbors within communication range.
    """
    agents_list = list(agents)
    for agent in agents_list:
        neighbors = get_neighbors(agent, agents_list)
        union_victims = set(agent.known_victims)
        for nb in neighbors:
            union_victims |= nb.known_victims
        agent.integrate_victim_info(union_victims)
        for nb in neighbors:
            nb.integrate_victim_info(union_victims)


# =========================
# Behavior (coverage)
# =========================
def choose_next_move(
    agent: Agent,
    world: World,
    occupied_positions: Set[Coord],
    rng: random.Random,
) -> Coord:
    """
    Very simple coverage behavior:
    - consider 4-neighborhood
    - prefer free cells that were not visited by this agent
    - if none available, move to any free neighbor
    - if stuck, stay in place
    """
    x, y = agent.position
    neighbors = world.get_neighbors_4(x, y)

    free_neighbors = [
        (nx, ny)
        for (nx, ny) in neighbors
        if world.is_free(nx, ny) and (nx, ny) not in occupied_positions
    ]

    if not free_neighbors:
        return agent.position  # stay put

    unvisited = [pos for pos in free_neighbors if pos not in agent.visited]
    if unvisited:
        return rng.choice(unvisited)

    return rng.choice(free_neighbors)


# =========================
# Perception
# =========================
def sense_victims_in_range(
    world: World,
    position: Coord,
    sensing_range: int,
) -> List[Coord]:
    """Returns all victim cells within a square neighborhood."""
    x0, y0 = position
    victims: List[Coord] = []

    for dx in range(-sensing_range, sensing_range + 1):
        for dy in range(-sensing_range, sensing_range + 1):
            x = x0 + dx
            y = y0 + dy
            if not world.in_bounds(x, y):
                continue
            if world.grid[y][x] == VICTIM:
                victims.append((x, y))

    return victims


def detect_victims(world: World, position: Coord, sensing_range: int) -> List[Coord]:
    """Wrapper for sensing (could add noise later)."""
    return sense_victims_in_range(world, position, sensing_range)


# =========================
# Simulation helpers
# =========================
def spawn_agents(
    world: World,
    num_agents: int,
    sensing_range: int,
    comm_range: int,
    rng: random.Random,
) -> List[Agent]:
    agents: List[Agent] = []
    occupied: Set[Coord] = set()

    while len(agents) < num_agents:
        x = rng.randrange(world.width)
        y = rng.randrange(world.height)
        if not world.is_free(x, y):
            continue
        if (x, y) in occupied:
            continue
        agent = Agent(
            agent_id=len(agents),
            position=(x, y),
            sensing_range=sensing_range,
            comm_range=comm_range,
        )
        agents.append(agent)
        occupied.add((x, y))

    return agents


def run_simulation() -> None:
    # ----- Configuration -----
    world_width = 30
    world_height = 20
    obstacle_density = 0.12
    num_victims = 10

    num_agents = 8
    comm_range = 5
    sensing_range = 3

    max_steps = 300
    seed = 42
    render_interval = 20  # show map every N steps (0 = never)
    # -------------------------

    rng = random.Random(seed)

    world = World(
        width=world_width,
        height=world_height,
        obstacle_density=obstacle_density,
        num_victims=num_victims,
        seed=seed,
    )

    agents = spawn_agents(
        world=world,
        num_agents=num_agents,
        sensing_range=sensing_range,
        comm_range=comm_range,
        rng=rng,
    )

    total_victims = len(world.victims)
    print(f"World: {world.width}x{world.height}")
    print(f"Obstacles: {len(world.obstacles)}, Victims: {total_victims}")
    print(f"Agents: {len(agents)}\n")

    start_time = time.time()
    step = 0

    while step < max_steps:
        step += 1

        # 1) Perception & victim detection
        for agent in agents:
            detected = detect_victims(
                world=world,
                position=agent.position,
                sensing_range=agent.sensing_range,
            )
            agent.integrate_victim_info(set(detected))

        # 2) Communication
        share_victim_information(agents)

        # 3) Movement decisions
        occupied_positions: Set[Coord] = {a.position for a in agents}
        new_positions: List[Coord] = []

        for agent in agents:
            other_occupied = occupied_positions - {agent.position}
            next_pos = choose_next_move(
                agent=agent,
                world=world,
                occupied_positions=other_occupied,
                rng=rng,
            )
            new_positions.append(next_pos)

        # 4) Apply movements
        for agent, new_pos in zip(agents, new_positions):
            agent.update_position(new_pos)

        # Metrics
        discovered_victims = set().union(*(a.known_victims for a in agents))
        num_found = len(discovered_victims)

        if render_interval and step % render_interval == 0:
            print(f"\n=== Step {step} ===")
            agent_positions = {a.position for a in agents}
            print(world.render_ascii(agent_positions))
            print(
                f"Discovered victims: {num_found}/{total_victims} | "
                f"Visited cells (agent 0): {len(agents[0].visited)}"
            )

        # Stop if all victims found
        if num_found >= total_victims and total_victims > 0:
            elapsed = time.time() - start_time
            print(f"\nAll victims discovered at step {step} (t={elapsed:.2f}s).")
            break

    else:
        elapsed = time.time() - start_time
        print(
            f"\nSimulation finished (max_steps={max_steps}). "
            f"Discovered victims: {num_found}/{total_victims}, "
            f"time={elapsed:.2f}s"
        )


def main() -> None:
    run_simulation()


if __name__ == "__main__":
    main()
