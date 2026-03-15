"""Pygame GUI for watching agents play or playing as human.

Supports all 4 environments (LineWorld, GridWorld, TicTacToe, Bobail).
Two modes: watch (AI vs AI) and play (human vs AI or human vs human).
"""

import sys
import time

import pygame
import numpy as np

from environments import ENV_REGISTRY, get_env
from environments.bobail import BOARD_SIZE as BOBAIL_SIZE, PHASE_BOBAIL, _idx_to_rc, _rc_to_idx
from agents import AGENT_REGISTRY, get_agent
from agents.human_agent import HumanAgent

# --- Colors ---
BG = (30, 30, 40)
PANEL_BG = (45, 45, 58)
WHITE = (240, 240, 240)
GRAY = (160, 160, 170)
DARK_GRAY = (80, 80, 90)
ACCENT = (100, 140, 255)
ACCENT_HOVER = (130, 165, 255)
GREEN = (80, 200, 120)
RED = (220, 80, 80)
YELLOW = (240, 200, 60)
ORANGE = (240, 150, 50)
BOARD_LIGHT = (220, 210, 195)
BOARD_DARK = (170, 155, 135)
CELL_HIGHLIGHT = (100, 200, 100, 120)

WINDOW_W, WINDOW_H = 900, 700
FPS = 60
AI_STEP_DELAY_MS = 400


def run():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("DRL Project — Game GUI")
    clock = pygame.time.Clock()

    app = App(screen, clock)
    app.main_loop()
    pygame.quit()


class Button:
    def __init__(self, rect: pygame.Rect, text: str, font: pygame.font.Font,
                 color=ACCENT, hover_color=ACCENT_HOVER, text_color=WHITE):
        self.rect = rect
        self.text = text
        self.font = font
        self.color = color
        self.hover_color = hover_color
        self.text_color = text_color

    def draw(self, surface: pygame.Surface, mouse_pos: tuple):
        hovered = self.rect.collidepoint(mouse_pos)
        color = self.hover_color if hovered else self.color
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        txt = self.font.render(self.text, True, self.text_color)
        txt_rect = txt.get_rect(center=self.rect.center)
        surface.blit(txt, txt_rect)

    def clicked(self, mouse_pos: tuple) -> bool:
        return self.rect.collidepoint(mouse_pos)


class App:
    STATE_MENU = 0
    STATE_PLAYING = 1
    STATE_GAME_OVER = 2

    def __init__(self, screen: pygame.Surface, clock: pygame.time.Clock):
        self.screen = screen
        self.clock = clock
        self.font_lg = pygame.font.SysFont("Arial", 28, bold=True)
        self.font_md = pygame.font.SysFont("Arial", 20)
        self.font_sm = pygame.font.SysFont("Arial", 16)

        self.state = self.STATE_MENU
        self.env = None
        self.env_name = ""
        self.agents = []  # [agent_p0, agent_p1] or [agent] for single-player
        self.agent_names = []

        # Menu selections
        self.env_choices = list(ENV_REGISTRY.keys())
        self.agent_choices = list(AGENT_REGISTRY.keys())
        self.selected_env = 0
        self.selected_agent0 = 0
        self.selected_agent1 = 0

        # Game state
        self.game_state = None
        self.done = False
        self.last_reward = 0.0
        self.winner_text = ""
        self.last_ai_step_time = 0
        self.selected_piece = None  # for Bobail click-based input
        self.valid_moves_from_selected = {}  # to_cell -> action_int

    def main_loop(self):
        running = True
        while running:
            mouse_pos = pygame.mouse.get_pos()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(mouse_pos)
                if event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)

            if self.state == self.STATE_PLAYING and not self.done:
                self._try_ai_step()

            self.screen.fill(BG)
            if self.state == self.STATE_MENU:
                self._draw_menu(mouse_pos)
            elif self.state in (self.STATE_PLAYING, self.STATE_GAME_OVER):
                self._draw_game(mouse_pos)

            pygame.display.flip()
            self.clock.tick(FPS)

    # --- Menu ---

    def _build_menu_buttons(self):
        """Rebuild menu buttons (called each frame for simplicity)."""
        buttons = {}
        cx = WINDOW_W // 2
        y = 80

        # Title
        buttons["_title_y"] = y
        y += 60

        # Env selector
        buttons["_env_label_y"] = y
        y += 30
        for i, name in enumerate(self.env_choices):
            r = pygame.Rect(cx - 150 + i * 80, y, 70, 32)
            color = GREEN if i == self.selected_env else DARK_GRAY
            buttons[f"env_{i}"] = Button(r, name.split("_")[-1] if "_" not in name else name.replace("_", " ").title(),
                                         self.font_sm, color=color)
        # Shorten labels
        short_env = ["Line", "Grid", "TicTac", "Bobail"]
        for i, label in enumerate(short_env):
            r = pygame.Rect(cx - 180 + i * 95, y, 85, 32)
            color = GREEN if i == self.selected_env else DARK_GRAY
            buttons[f"env_{i}"] = Button(r, label, self.font_sm, color=color)
        y += 55

        # Agent 0 selector
        buttons["_agent0_label_y"] = y
        y += 30
        for i, name in enumerate(self.agent_choices):
            r = pygame.Rect(cx - 100 + i * 110, y, 100, 32)
            color = GREEN if i == self.selected_agent0 else DARK_GRAY
            buttons[f"agent0_{i}"] = Button(r, name.title(), self.font_sm, color=color)
        y += 55

        # Agent 1 selector (only for adversarial envs)
        env_name = self.env_choices[self.selected_env]
        test_env = get_env(env_name)
        if test_env.is_adversarial():
            buttons["_agent1_label_y"] = y
            y += 30
            for i, name in enumerate(self.agent_choices):
                r = pygame.Rect(cx - 100 + i * 110, y, 100, 32)
                color = GREEN if i == self.selected_agent1 else DARK_GRAY
                buttons[f"agent1_{i}"] = Button(r, name.title(), self.font_sm, color=color)
            y += 55

        # Start button
        buttons["start"] = Button(
            pygame.Rect(cx - 80, y + 10, 160, 45), "Start", self.font_lg, color=ACCENT)

        return buttons

    def _draw_menu(self, mouse_pos):
        self._menu_buttons = self._build_menu_buttons()
        cx = WINDOW_W // 2

        title = self.font_lg.render("Deep RL — Game GUI", True, WHITE)
        self.screen.blit(title, title.get_rect(center=(cx, self._menu_buttons["_title_y"])))

        env_label = self.font_md.render("Environment", True, GRAY)
        self.screen.blit(env_label, env_label.get_rect(center=(cx, self._menu_buttons["_env_label_y"])))

        a0_label = self.font_md.render("Agent (Player 1)", True, GRAY)
        self.screen.blit(a0_label, a0_label.get_rect(center=(cx, self._menu_buttons["_agent0_label_y"])))

        if "_agent1_label_y" in self._menu_buttons:
            a1_label = self.font_md.render("Agent (Player 2)", True, GRAY)
            self.screen.blit(a1_label, a1_label.get_rect(center=(cx, self._menu_buttons["_agent1_label_y"])))

        for key, val in self._menu_buttons.items():
            if isinstance(val, Button):
                val.draw(self.screen, mouse_pos)

    def _handle_menu_click(self, mouse_pos):
        btns = self._menu_buttons
        for key, val in btns.items():
            if not isinstance(val, Button):
                continue
            if not val.clicked(mouse_pos):
                continue
            if key.startswith("env_"):
                self.selected_env = int(key.split("_")[1])
            elif key.startswith("agent0_"):
                self.selected_agent0 = int(key.split("_")[1])
            elif key.startswith("agent1_"):
                self.selected_agent1 = int(key.split("_")[1])
            elif key == "start":
                self._start_game()

    def _start_game(self):
        self.env_name = self.env_choices[self.selected_env]
        self.env = get_env(self.env_name)

        a0_name = self.agent_choices[self.selected_agent0]
        agent0 = get_agent(a0_name, self.env)
        self.agent_names = [a0_name]

        if self.env.is_adversarial():
            a1_name = self.agent_choices[self.selected_agent1]
            agent1 = get_agent(a1_name, self.env)
            self.agents = [agent0, agent1]
            self.agent_names.append(a1_name)
        else:
            self.agents = [agent0]

        self.game_state = self.env.reset()
        self.done = False
        self.last_reward = 0.0
        self.winner_text = ""
        self.last_ai_step_time = pygame.time.get_ticks()
        self.selected_piece = None
        self.valid_moves_from_selected = {}
        self.state = self.STATE_PLAYING

    # --- Game logic ---

    def _current_agent(self) -> "Agent":
        player = self.env.current_player()
        return self.agents[min(player, len(self.agents) - 1)]

    def _is_human_turn(self) -> bool:
        return isinstance(self._current_agent(), HumanAgent)

    def _try_ai_step(self):
        if self._is_human_turn():
            return
        now = pygame.time.get_ticks()
        if now - self.last_ai_step_time < AI_STEP_DELAY_MS:
            return
        self._do_step_ai()
        self.last_ai_step_time = now

    def _do_step_ai(self):
        agent = self._current_agent()
        state = self.env.state_description()
        available = self.env.available_actions()
        if not available:
            self.done = True
            return
        action = agent.act(state, available)
        self.game_state, self.last_reward, self.done = self.env.step(action)
        if self.done:
            self._set_game_over()

    def _do_step_human(self, action: int):
        agent = self._current_agent()
        agent.set_action(action)
        state = self.env.state_description()
        available = self.env.available_actions()
        _ = agent.act(state, available)
        self.game_state, self.last_reward, self.done = self.env.step(action)
        self.selected_piece = None
        self.valid_moves_from_selected = {}
        self.last_ai_step_time = pygame.time.get_ticks()
        if self.done:
            self._set_game_over()

    def _set_game_over(self):
        self.state = self.STATE_GAME_OVER
        if self.last_reward > 0:
            # The player who just acted won; but current_player already switched
            winner = 1 - self.env.current_player() if self.env.is_adversarial() else 0
            self.winner_text = f"Player {winner + 1} wins!"
        elif self.last_reward < 0:
            winner = self.env.current_player()
            self.winner_text = f"Player {winner + 1} wins!"
        else:
            if self.env.is_adversarial():
                self.winner_text = "Draw!"
            else:
                self.winner_text = "Game over!"

    # --- Input handling ---

    def _handle_click(self, mouse_pos):
        if self.state == self.STATE_MENU:
            self._handle_menu_click(mouse_pos)
        elif self.state == self.STATE_PLAYING:
            self._handle_game_click(mouse_pos)
        elif self.state == self.STATE_GAME_OVER:
            self._handle_gameover_click(mouse_pos)

    def _handle_key(self, key):
        if self.state == self.STATE_PLAYING and self._is_human_turn():
            action = self._key_to_action(key)
            if action is not None and action in self.env.available_actions():
                self._do_step_human(action)

    def _key_to_action(self, key) -> int | None:
        """Map keyboard to action for LineWorld/GridWorld human play."""
        if self.env_name == "line_world":
            if key == pygame.K_LEFT:
                return 0
            if key == pygame.K_RIGHT:
                return 1
        elif self.env_name == "grid_world":
            mapping = {pygame.K_UP: 0, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 3}
            return mapping.get(key)
        return None

    def _handle_game_click(self, mouse_pos):
        if not self._is_human_turn():
            return
        if self.env_name == "tictactoe":
            self._handle_tictactoe_click(mouse_pos)
        elif self.env_name == "bobail":
            self._handle_bobail_click(mouse_pos)

    def _handle_gameover_click(self, mouse_pos):
        if hasattr(self, "_back_btn") and self._back_btn.clicked(mouse_pos):
            self.state = self.STATE_MENU

    # --- TicTacToe click ---

    def _handle_tictactoe_click(self, mouse_pos):
        board_rect = self._get_board_rect(3, 3)
        cell_w = board_rect.width // 3
        cell_h = board_rect.height // 3
        mx, my = mouse_pos
        if not board_rect.collidepoint(mx, my):
            return
        col = (mx - board_rect.x) // cell_w
        row = (my - board_rect.y) // cell_h
        action = row * 3 + col
        if action in self.env.available_actions():
            self._do_step_human(action)

    # --- Bobail click ---

    def _handle_bobail_click(self, mouse_pos):
        board_rect = self._get_board_rect(BOBAIL_SIZE, BOBAIL_SIZE)
        cell_w = board_rect.width // BOBAIL_SIZE
        cell_h = board_rect.height // BOBAIL_SIZE
        mx, my = mouse_pos
        if not board_rect.collidepoint(mx, my):
            return
        col = (mx - board_rect.x) // cell_w
        row = (my - board_rect.y) // cell_h
        clicked_idx = _rc_to_idx(row, col)

        available = self.env.available_actions()

        if self.selected_piece is not None:
            # Second click: try to move to this cell
            if clicked_idx in self.valid_moves_from_selected:
                action = self.valid_moves_from_selected[clicked_idx]
                self._do_step_human(action)
                return
            # Clicked elsewhere: deselect and fall through to try selecting
            self.selected_piece = None
            self.valid_moves_from_selected = {}

        # First click: select a source piece
        moves_from_here = {}
        for a in available:
            from_cell = a // (BOBAIL_SIZE * BOBAIL_SIZE)
            to_cell = a % (BOBAIL_SIZE * BOBAIL_SIZE)
            if from_cell == clicked_idx:
                moves_from_here[to_cell] = a

        if moves_from_here:
            self.selected_piece = clicked_idx
            self.valid_moves_from_selected = moves_from_here

    # --- Drawing: Game ---

    def _get_board_rect(self, rows, cols) -> pygame.Rect:
        board_area_w = min(500, WINDOW_W - 250)
        board_area_h = min(500, WINDOW_H - 120)
        cell = min(board_area_w // cols, board_area_h // rows)
        bw = cell * cols
        bh = cell * rows
        bx = (WINDOW_W - 150) // 2 - bw // 2 + 30
        by = (WINDOW_H - 50) // 2 - bh // 2 + 25
        return pygame.Rect(bx, by, bw, bh)

    def _draw_game(self, mouse_pos):
        dispatch = {
            "line_world": self._draw_line_world,
            "grid_world": self._draw_grid_world,
            "tictactoe": self._draw_tictactoe,
            "bobail": self._draw_bobail,
        }
        dispatch[self.env_name]()
        self._draw_sidebar(mouse_pos)

    def _draw_sidebar(self, mouse_pos):
        panel_x = WINDOW_W - 210
        pygame.draw.rect(self.screen, PANEL_BG, (panel_x, 0, 210, WINDOW_H))

        y = 20
        title = self.font_md.render(self.env_name.replace("_", " ").title(), True, WHITE)
        self.screen.blit(title, (panel_x + 15, y))
        y += 35

        player = self.env.current_player()
        for i, name in enumerate(self.agent_names):
            color = ACCENT if i == player and not self.done else GRAY
            label = f"P{i+1}: {name}"
            txt = self.font_sm.render(label, True, color)
            self.screen.blit(txt, (panel_x + 15, y))
            y += 22

        y += 15
        if self.done:
            wt = self.font_md.render(self.winner_text, True, YELLOW)
            self.screen.blit(wt, (panel_x + 15, y))
            y += 40
        elif self._is_human_turn():
            hint = self.font_sm.render("Your turn", True, GREEN)
            self.screen.blit(hint, (panel_x + 15, y))
            if self.env_name in ("line_world", "grid_world"):
                hint2 = self.font_sm.render("Use arrow keys", True, GRAY)
                self.screen.blit(hint2, (panel_x + 15, y + 20))
            elif self.env_name == "bobail":
                phase_text = "Move Bobail" if self.env._phase == PHASE_BOBAIL else "Move a piece"
                hint2 = self.font_sm.render(phase_text, True, GRAY)
                self.screen.blit(hint2, (panel_x + 15, y + 20))
            y += 45

        # Back to menu button
        self._back_btn = Button(
            pygame.Rect(panel_x + 20, WINDOW_H - 60, 170, 40),
            "Back to Menu", self.font_sm, color=DARK_GRAY)
        self._back_btn.draw(self.screen, mouse_pos)

    # --- LineWorld ---

    def _draw_line_world(self):
        env = self.env
        size = env._size
        pos = env._pos

        cell_w = min(80, (WINDOW_W - 300) // size)
        cell_h = 60
        total_w = cell_w * size
        start_x = (WINDOW_W - 210) // 2 - total_w // 2
        start_y = WINDOW_H // 2 - cell_h // 2

        for i in range(size):
            x = start_x + i * cell_w
            color = BOARD_LIGHT if i % 2 == 0 else BOARD_DARK
            if i == pos:
                color = ACCENT
            elif i == size - 1:
                color = GREEN
            pygame.draw.rect(self.screen, color, (x, start_y, cell_w, cell_h))
            pygame.draw.rect(self.screen, BG, (x, start_y, cell_w, cell_h), 2)

            if i == pos:
                txt = self.font_lg.render("A", True, WHITE)
                self.screen.blit(txt, txt.get_rect(center=(x + cell_w // 2, start_y + cell_h // 2)))
            elif i == size - 1:
                txt = self.font_lg.render("G", True, WHITE)
                self.screen.blit(txt, txt.get_rect(center=(x + cell_w // 2, start_y + cell_h // 2)))

    # --- GridWorld ---

    def _draw_grid_world(self):
        env = self.env
        rows, cols = env._rows, env._cols
        board_rect = self._get_board_rect(rows, cols)
        cell_w = board_rect.width // cols
        cell_h = board_rect.height // rows

        for r in range(rows):
            for c in range(cols):
                x = board_rect.x + c * cell_w
                y = board_rect.y + r * cell_h
                color = BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK
                if r == env._row and c == env._col:
                    color = ACCENT
                elif r == rows - 1 and c == cols - 1:
                    color = GREEN
                pygame.draw.rect(self.screen, color, (x, y, cell_w, cell_h))
                pygame.draw.rect(self.screen, BG, (x, y, cell_w, cell_h), 1)

                if r == env._row and c == env._col:
                    txt = self.font_lg.render("A", True, WHITE)
                    self.screen.blit(txt, txt.get_rect(center=(x + cell_w // 2, y + cell_h // 2)))
                elif r == rows - 1 and c == cols - 1:
                    txt = self.font_lg.render("G", True, WHITE)
                    self.screen.blit(txt, txt.get_rect(center=(x + cell_w // 2, y + cell_h // 2)))

    # --- TicTacToe ---

    def _draw_tictactoe(self):
        env = self.env
        board_rect = self._get_board_rect(3, 3)
        cell_w = board_rect.width // 3
        cell_h = board_rect.height // 3

        for r in range(3):
            for c in range(3):
                x = board_rect.x + c * cell_w
                y = board_rect.y + r * cell_h
                color = BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK
                pygame.draw.rect(self.screen, color, (x, y, cell_w, cell_h))
                pygame.draw.rect(self.screen, BG, (x, y, cell_w, cell_h), 2)

                idx = r * 3 + c
                mark = env._board[idx]
                if mark == 1:
                    self._draw_x(x, y, cell_w, cell_h)
                elif mark == 2:
                    self._draw_o(x, y, cell_w, cell_h)

    def _draw_x(self, x, y, w, h):
        margin = min(w, h) // 5
        color = ACCENT
        pygame.draw.line(self.screen, color, (x + margin, y + margin),
                         (x + w - margin, y + h - margin), 4)
        pygame.draw.line(self.screen, color, (x + w - margin, y + margin),
                         (x + margin, y + h - margin), 4)

    def _draw_o(self, x, y, w, h):
        margin = min(w, h) // 5
        color = RED
        cx = x + w // 2
        cy = y + h // 2
        radius = min(w, h) // 2 - margin
        pygame.draw.circle(self.screen, color, (cx, cy), radius, 4)

    # --- Bobail ---

    def _draw_bobail(self):
        env = self.env
        board_rect = self._get_board_rect(BOBAIL_SIZE, BOBAIL_SIZE)
        cell_w = board_rect.width // BOBAIL_SIZE
        cell_h = board_rect.height // BOBAIL_SIZE

        for r in range(BOBAIL_SIZE):
            for c in range(BOBAIL_SIZE):
                x = board_rect.x + c * cell_w
                y = board_rect.y + r * cell_h
                color = BOARD_LIGHT if (r + c) % 2 == 0 else BOARD_DARK
                idx = _rc_to_idx(r, c)

                # Highlight valid move destinations
                if idx in self.valid_moves_from_selected:
                    color = (140, 210, 140)

                pygame.draw.rect(self.screen, color, (x, y, cell_w, cell_h))
                pygame.draw.rect(self.screen, BG, (x, y, cell_w, cell_h), 1)

                cx_pos = x + cell_w // 2
                cy_pos = y + cell_h // 2
                piece_r = min(cell_w, cell_h) // 3

                if idx == env._bobail:
                    pygame.draw.circle(self.screen, YELLOW, (cx_pos, cy_pos), piece_r)
                    pygame.draw.circle(self.screen, BG, (cx_pos, cy_pos), piece_r, 2)
                elif idx in env._pieces[0]:
                    sel = (self.selected_piece == idx)
                    c0 = ORANGE if sel else ACCENT
                    pygame.draw.circle(self.screen, c0, (cx_pos, cy_pos), piece_r)
                    pygame.draw.circle(self.screen, BG, (cx_pos, cy_pos), piece_r, 2)
                    txt = self.font_sm.render("1", True, WHITE)
                    self.screen.blit(txt, txt.get_rect(center=(cx_pos, cy_pos)))
                elif idx in env._pieces[1]:
                    sel = (self.selected_piece == idx)
                    c1 = ORANGE if sel else RED
                    pygame.draw.circle(self.screen, c1, (cx_pos, cy_pos), piece_r)
                    pygame.draw.circle(self.screen, BG, (cx_pos, cy_pos), piece_r, 2)
                    txt = self.font_sm.render("2", True, WHITE)
                    self.screen.blit(txt, txt.get_rect(center=(cx_pos, cy_pos)))

        # Draw home row indicators
        for c in range(BOBAIL_SIZE):
            # P1 home = row 4 (bottom), P2 home = row 0 (top)
            x_top = board_rect.x + c * cell_w + cell_w // 2
            x_bot = x_top
            txt_top = self.font_sm.render("^", True, (RED[0], RED[1], RED[2], 80))
            txt_bot = self.font_sm.render("v", True, (ACCENT[0], ACCENT[1], ACCENT[2], 80))
            self.screen.blit(txt_top, txt_top.get_rect(center=(x_top, board_rect.y - 12)))
            self.screen.blit(txt_bot, txt_bot.get_rect(center=(x_bot, board_rect.y + board_rect.height + 12)))
