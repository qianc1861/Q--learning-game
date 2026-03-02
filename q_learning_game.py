import random
from typing import Dict, Tuple, List

import pygame


Position = Tuple[int, int]
Action = str


class GridWorld:
    """
    A simple 2D grid-world for Q-learning.

    Legend:
    - S: start
    - G: goal (+1 reward, terminal)
    - X: trap (-1 reward, terminal)
    - .: empty
    """

    def __init__(self, width: int = 10, height: int = 10):
        self.width = width
        self.height = height

        self.start: Position = (0, 0)
        self.goal: Position = (4, 4)
        self.traps: List[Position] = []

        self.step_reward = -0.04
        self.goal_reward = 1.0
        self.trap_reward = -1.0

        self.actions: List[Action] = ["up", "down", "left", "right"]

        self.agent_pos: Position = self.start

        # Randomize map layout each time the environment is created
        self.randomize_layout()

    def randomize_layout(self, num_traps: int | None = None) -> None:
        """Randomly choose goal and trap positions so that every map is different."""
        if num_traps is None:
            # Automatically set number of traps based on map size to make paths more complex
            num_traps = max(5, (self.width * self.height) // 8)
        all_cells = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) != self.start
        ]
        self.goal = random.choice(all_cells)
        remaining = [c for c in all_cells if c != self.goal]
        if num_traps > len(remaining):
            num_traps = len(remaining)
        self.traps = random.sample(remaining, k=num_traps)

    def reset(self) -> Position:
        self.agent_pos = self.start
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Position, float, bool]:
        x, y = self.agent_pos

        if action == "up":
            y = max(0, y - 1)
        elif action == "down":
            y = min(self.height - 1, y + 1)
        elif action == "left":
            x = max(0, x - 1)
        elif action == "right":
            x = min(self.width - 1, x + 1)

        new_pos = (x, y)
        self.agent_pos = new_pos

        reward = self.step_reward
        done = False

        if new_pos == self.goal:
            reward = self.goal_reward
            done = True
        elif new_pos in self.traps:
            reward = self.trap_reward
            done = True

        return new_pos, reward, done

    def render(self, agent_pos: Position | None = None) -> None:
        if agent_pos is None:
            agent_pos = self.agent_pos

        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == agent_pos:
                    row.append("A")
                elif pos == self.start:
                    row.append("S")
                elif pos == self.goal:
                    row.append("G")
                elif pos in self.traps:
                    row.append("X")
                else:
                    row.append(".")
            print(" ".join(row))
        print()


class QLearningAgent:
    def __init__(
        self,
        actions: List[Action],
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        epsilon: float = 0.2,
    ):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table: Dict[Tuple[Position, Action], float] = {}

    def get_q(self, state: Position, action: Action) -> float:
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state: Position, explore: bool = True) -> Action:
        if explore and random.random() < self.epsilon:
            return random.choice(self.actions)

        q_values = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        best_actions = [
            a for a, q in zip(self.actions, q_values) if q == max_q
        ]
        return random.choice(best_actions)

    def learn(
        self,
        state: Position,
        action: Action,
        reward: float,
        next_state: Position,
        done: bool,
    ) -> None:
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_qs = [self.get_q(next_state, a) for a in self.actions]
            target = reward + self.gamma * max(next_qs)

        new_q = current_q + self.lr * (target - current_q)
        self.q_table[(state, action)] = new_q


class PygameViewer:
    def __init__(self, env: GridWorld, cell_size: int = 60):
        self.env = env
        self.cell_size = cell_size
        self.margin = 40
        self.width = env.width * cell_size + self.margin * 2
        self.height = env.height * cell_size + self.margin * 2 + 60

        pygame.init()
        pygame.display.set_caption("Q-learning GridWorld")
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 20)
        self.title_font = pygame.font.SysFont("Arial", 48, bold=True)

        self.colors = {
            "bg": (30, 30, 30),
            "grid": (200, 200, 200),
            "start": (150, 150, 255),
            "goal": (80, 200, 120),
            "trap": (220, 80, 80),
            "agent": (70, 160, 255),
            "empty": (245, 245, 245),
        }

    def draw_grid(self, agent_pos: Position) -> None:
        self.screen.fill(self.colors["bg"])

        for y in range(self.env.height):
            for x in range(self.env.width):
                px = self.margin + x * self.cell_size
                py = self.margin + y * self.cell_size
                rect = pygame.Rect(px, py, self.cell_size, self.cell_size)

                pos = (x, y)
                color = self.colors["empty"]
                if pos == self.env.start:
                    color = self.colors["start"]
                if pos == self.env.goal:
                    color = self.colors["goal"]
                if pos in self.env.traps:
                    color = self.colors["trap"]
                if pos == agent_pos:
                    color = self.colors["agent"]

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, self.colors["grid"], rect, 2)

        info_y = self.margin + self.env.height * self.cell_size + 10
        text = self.font.render("Press ESC to close the window", True, (230, 230, 230))
        self.screen.blit(text, (self.margin, info_y))

    def show_start_screen(self) -> bool:
        """Show a Tank-War-like start screen. Return False if user quits."""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key in (pygame.K_y, pygame.K_RETURN, pygame.K_SPACE):
                        return True

            self.screen.fill((255, 255, 255))

            title_text = self.title_font.render("Q-learning Grid War", True, (0, 0, 0))
            title_rect = title_text.get_rect(center=(self.width // 2, self.height // 2 - 120))
            self.screen.blit(title_text, title_rect)

            lines = [
                "In this game, an AI agent",
                "learns to navigate a random grid world.",
                "It will try to reach the goal",
                "while avoiding deadly traps.",
                "",
                "Press Y to start demo",
                "Press R to replay after a run",
                "Press ESC to quit the game",
            ]
            y = self.height // 2 - 40
            for line in lines:
                text = self.font.render(line, True, (0, 120, 0))
                rect = text.get_rect(center=(self.width // 2, y))
                self.screen.blit(text, rect)
                y += 28

            pygame.display.flip()
            self.clock.tick(60)

    def run_episode(
        self,
        agent: QLearningAgent,
        max_steps: int = 30,
        delay_ms: int = 400,
    ) -> None:
        state = self.env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return

            action = agent.choose_action(state, explore=False)
            next_state, _, done = self.env.step(action)
            state = next_state
            steps += 1

            self.draw_grid(agent_pos=state)
            pygame.display.flip()
            self.clock.tick(1000 // max(delay_ms, 1))


def run_pygame_demo(
    episodes: int = 2000,
    max_steps: int = 80,
    max_steps_per_episode: int = 200,
) -> None:
    # Use a temporary environment just for the start screen
    temp_env = GridWorld()
    temp_viewer = PygameViewer(temp_env)
    start = temp_viewer.show_start_screen()
    if not start:
        pygame.quit()
        return

    running = True
    while running:
        # Before each replay, train a brand-new environment so the map is different
        env, agent = train_agent(
            episodes=episodes,
            max_steps_per_episode=max_steps_per_episode,
        )
        viewer = PygameViewer(env)
        viewer.run_episode(agent, max_steps=max_steps)

        # 每次自动演示结束后，等待玩家选择是否重放或退出
        waiting = True
        while waiting and running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    waiting = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        waiting = False
                    if event.key in (pygame.K_r, pygame.K_y, pygame.K_SPACE):
                        # 再来一轮演示：跳出等待，重新训练新的地图
                        waiting = False

            viewer.draw_grid(agent_pos=env.agent_pos)
            info_y = viewer.margin + env.height * viewer.cell_size + 35
            hint = viewer.font.render(
                "Press R/Y/SPACE to replay (new map), ESC to quit",
                True,
                (230, 230, 230),
            )
            viewer.screen.blit(hint, (viewer.margin, info_y))
            pygame.display.flip()
            viewer.clock.tick(60)

    pygame.quit()


def train_agent(
    episodes: int = 2000,
    max_steps_per_episode: int = 200,
    width: int = 10,
    height: int = 10,
) -> Tuple[GridWorld, QLearningAgent]:
    env = GridWorld(width=width, height=height)
    agent = QLearningAgent(actions=env.actions)

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0

        for _ in range(max_steps_per_episode):
            action = agent.choose_action(state, explore=True)
            next_state, reward, done = env.step(action)

            agent.learn(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        if (ep + 1) % 200 == 0:
            print(
                f"Episode {ep + 1}/{episodes}, "
                f"total reward: {total_reward:.2f}"
            )

    return env, agent


def play_with_trained_agent(
    env: GridWorld,
    agent: QLearningAgent,
    max_steps: int = 30,
) -> None:
    state = env.reset()
    print("Run one demo with the learned policy:")
    env.render()

    for step in range(max_steps):
        action = agent.choose_action(state, explore=False)
        next_state, reward, done = env.step(action)

        print(
            f"Step {step + 1}: action = {action}, "
            f"reward = {reward:.2f}"
        )
        env.render()

        state = next_state

        if done:
            if state == env.goal:
                print("Reached the goal!")
            elif state in env.traps:
                print("Fell into a trap!")
            break
    else:
        print("Episode did not finish within the step limit.")


def interactive_loop() -> None:
    print("=== Q-learning Grid Game ===")
    print("Environment: 10x10 grid, start S, goal G, traps X.")
    print("The agent learns a policy with Q-learning to reach G and avoid traps.")
    print()

    while True:
        try:
            episodes = int(
                input("Please enter number of training episodes (recommended 1000-3000, default 2000): ") or "2000"
            )
        except ValueError:
            print("Invalid input, using default 2000 episodes.")
            episodes = 2000

        mode = input("Choose demo mode: 1=terminal print, 2=Pygame visualization (default 2): ").strip()
        if mode == "1":
            env, agent = train_agent(episodes=episodes)
            play_with_trained_agent(env, agent)
        else:
            try:
                run_pygame_demo(episodes=episodes)
            except pygame.error as exc:
                print(f"Pygame error: {exc}")
                print("Falling back to terminal print mode.")
                env, agent = train_agent(episodes=episodes)
                play_with_trained_agent(env, agent)

        again = input("Train and demo again? (y/n): ").strip().lower()
        if again != "y":
            print("Game over, thanks for playing!")
            break


if __name__ == "__main__":
    interactive_loop()

