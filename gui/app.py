"""Interface graphique Pygame pour observer les agents jouer ou jouer en tant qu'humain.

Supporte les 4 environnements (LineWorld, GridWorld, TicTacToe, Bobail).
Deux modes : observer (IA vs IA) et jouer (humain vs IA ou humain vs humain).
"""

import glob
import os
import re
import sys
import time

import pygame
import numpy as np
import yaml

from environments import ENV_REGISTRY, get_env
from environments.bobail import BOARD_SIZE as BOBAIL_SIZE, PHASE_BOBAIL, _idx_to_rc, _rc_to_idx
from agents import AGENT_REGISTRY, get_agent
from agents.human_agent import HumanAgent

# --- Couleurs ---
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

AGENTS_WITHOUT_MODELS = {"random", "human"}
MAX_MODEL_BUTTONS = 4


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
        self.agents = []  # [agent_j0, agent_j1] ou [agent] pour un joueur
        self.agent_names = []

        # Selections du menu
        self.env_choices = list(ENV_REGISTRY.keys())
        self.agent_choices = list(AGENT_REGISTRY.keys())
        self.selected_env = 0
        self.selected_agent0 = 0
        self.selected_agent1 = 0

        # Selections de modele (index dans la liste des choix de modeles)
        self.selected_model0 = 0
        self.selected_model1 = 0
        self._model_scroll0 = 0
        self._model_scroll1 = 0
        self._cached_model_choices: dict[tuple, list] = {}

        # Etat de la partie
        self.game_state = None
        self.done = False
        self.last_reward = 0.0
        self.winner_text = ""
        self.last_ai_step_time = 0
        self.selected_piece = None  # pour la saisie par clic dans Bobail
        self.valid_moves_from_selected = {}  # case_arrivee -> action_int

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

    def _build_model_row(self, buttons, cx: int, y: int, prefix: str,
                         choices: list, selected: int, scroll: int) -> int:
        """Ajoute les boutons de selection de modele au dict buttons. Retourne y mis a jour."""
        buttons[f"_{prefix}_label_y"] = y
        y += 25

        if not choices:
            buttons[f"_{prefix}_no_models_y"] = y
            y += 35
            return y

        n_total = len(choices)
        n_visible = min(n_total, MAX_MODEL_BUTTONS)
        btn_w = 90
        gap = 5
        arrow_w = 25
        has_left = scroll > 0
        has_right = scroll + MAX_MODEL_BUTTONS < n_total

        total_w = n_visible * btn_w + max(0, n_visible - 1) * gap
        if has_left:
            total_w += arrow_w + gap
        if has_right:
            total_w += arrow_w + gap

        x = cx - total_w // 2

        if has_left:
            r = pygame.Rect(x, y, arrow_w, 28)
            buttons[f"{prefix}_left"] = Button(r, "<", self.font_sm, color=DARK_GRAY)
            x += arrow_w + gap

        for i in range(n_visible):
            idx = scroll + i
            if idx >= n_total:
                break
            label = choices[idx][0]
            r = pygame.Rect(x, y, btn_w, 28)
            color = GREEN if idx == selected else DARK_GRAY
            buttons[f"{prefix}_{i}"] = Button(r, label, self.font_sm, color=color)
            x += btn_w + gap

        if has_right:
            r = pygame.Rect(x, y, arrow_w, 28)
            buttons[f"{prefix}_right"] = Button(r, ">", self.font_sm, color=DARK_GRAY)

        y += 38
        return y

    def _build_menu_buttons(self):
        """Reconstruit les boutons du menu (appele a chaque frame par simplicite)."""
        buttons = {}
        cx = WINDOW_W // 2
        y = 80

        buttons["_title_y"] = y
        y += 60

        # Selecteur d'environnement
        buttons["_env_label_y"] = y
        y += 30
        short_env = ["Line", "Grid", "TicTac", "Bobail"]
        for i, label in enumerate(short_env):
            r = pygame.Rect(cx - 180 + i * 95, y, 85, 32)
            color = GREEN if i == self.selected_env else DARK_GRAY
            buttons[f"env_{i}"] = Button(r, label, self.font_sm, color=color)
        y += 50

        env_name = self.env_choices[self.selected_env]
        test_env = get_env(env_name)

        # Selecteur agent 0
        buttons["_agent0_label_y"] = y
        y += 28
        n_agents = len(self.agent_choices)
        agent_btn_w = 85
        agent_gap = 5
        agent_total_w = n_agents * agent_btn_w + (n_agents - 1) * agent_gap
        agent_x = cx - agent_total_w // 2
        for i, name in enumerate(self.agent_choices):
            r = pygame.Rect(agent_x + i * (agent_btn_w + agent_gap), y,
                            agent_btn_w, 28)
            color = GREEN if i == self.selected_agent0 else DARK_GRAY
            buttons[f"agent0_{i}"] = Button(r, name, self.font_sm, color=color)
        y += 40

        # Selecteur modele 0
        a0_name = self.agent_choices[self.selected_agent0]
        if self._agent_needs_model(a0_name):
            choices0 = self._get_model_choices(a0_name, env_name)
            y = self._build_model_row(buttons, cx, y, "model0",
                                      choices0, self.selected_model0,
                                      self._model_scroll0)

        # Agent 1 + Modele 1 (adversariel uniquement)
        if test_env.is_adversarial():
            buttons["_agent1_label_y"] = y
            y += 28
            for i, name in enumerate(self.agent_choices):
                r = pygame.Rect(agent_x + i * (agent_btn_w + agent_gap), y,
                                agent_btn_w, 28)
                color = GREEN if i == self.selected_agent1 else DARK_GRAY
                buttons[f"agent1_{i}"] = Button(r, name, self.font_sm, color=color)
            y += 40

            a1_name = self.agent_choices[self.selected_agent1]
            if self._agent_needs_model(a1_name):
                choices1 = self._get_model_choices(a1_name, env_name)
                y = self._build_model_row(buttons, cx, y, "model1",
                                          choices1, self.selected_model1,
                                          self._model_scroll1)

        # Bouton demarrer
        can_start = self._can_start()
        start_color = ACCENT if can_start else DARK_GRAY
        buttons["start"] = Button(
            pygame.Rect(cx - 80, y + 10, 160, 45), "Start", self.font_lg,
            color=start_color)

        return buttons

    def _draw_menu(self, mouse_pos):
        self._menu_buttons = self._build_menu_buttons()
        btns = self._menu_buttons
        cx = WINDOW_W // 2

        title = self.font_lg.render("Deep RL — Game GUI", True, WHITE)
        self.screen.blit(title, title.get_rect(center=(cx, btns["_title_y"])))

        env_label = self.font_md.render("Environment", True, GRAY)
        self.screen.blit(env_label, env_label.get_rect(center=(cx, btns["_env_label_y"])))

        a0_label = self.font_md.render("Agent (Player 1)", True, GRAY)
        self.screen.blit(a0_label, a0_label.get_rect(center=(cx, btns["_agent0_label_y"])))

        if "_model0_label_y" in btns:
            txt = self.font_sm.render("Model (P1)", True, GRAY)
            self.screen.blit(txt, txt.get_rect(center=(cx, btns["_model0_label_y"])))
        if "_model0_no_models_y" in btns:
            txt = self.font_sm.render("No trained models", True, RED)
            self.screen.blit(txt, txt.get_rect(center=(cx, btns["_model0_no_models_y"])))

        if "_agent1_label_y" in btns:
            a1_label = self.font_md.render("Agent (Player 2)", True, GRAY)
            self.screen.blit(a1_label, a1_label.get_rect(center=(cx, btns["_agent1_label_y"])))

        if "_model1_label_y" in btns:
            txt = self.font_sm.render("Model (P2)", True, GRAY)
            self.screen.blit(txt, txt.get_rect(center=(cx, btns["_model1_label_y"])))
        if "_model1_no_models_y" in btns:
            txt = self.font_sm.render("No trained models", True, RED)
            self.screen.blit(txt, txt.get_rect(center=(cx, btns["_model1_no_models_y"])))

        for key, val in btns.items():
            if isinstance(val, Button):
                val.draw(self.screen, mouse_pos)

    def _handle_model_click(self, key: str, prefix: str, agent_name: str,
                            env_name: str):
        """Gere les clics sur les boutons de selection de modele."""
        suffix = key[len(prefix) + 1:]
        if suffix == "left":
            if prefix == "model0":
                self._model_scroll0 = max(0, self._model_scroll0 - 1)
            else:
                self._model_scroll1 = max(0, self._model_scroll1 - 1)
        elif suffix == "right":
            choices = self._get_model_choices(agent_name, env_name)
            max_scroll = max(0, len(choices) - MAX_MODEL_BUTTONS)
            if prefix == "model0":
                self._model_scroll0 = min(self._model_scroll0 + 1, max_scroll)
            else:
                self._model_scroll1 = min(self._model_scroll1 + 1, max_scroll)
        else:
            vis_idx = int(suffix)
            scroll = self._model_scroll0 if prefix == "model0" else self._model_scroll1
            if prefix == "model0":
                self.selected_model0 = scroll + vis_idx
            else:
                self.selected_model1 = scroll + vis_idx

    def _handle_menu_click(self, mouse_pos):
        btns = self._menu_buttons
        env_name = self.env_choices[self.selected_env]

        for key, val in btns.items():
            if not isinstance(val, Button):
                continue
            if not val.clicked(mouse_pos):
                continue

            if key.startswith("env_"):
                new_idx = int(key.split("_")[1])
                if new_idx != self.selected_env:
                    self.selected_env = new_idx
                    self.selected_model0 = 0
                    self.selected_model1 = 0
                    self._model_scroll0 = 0
                    self._model_scroll1 = 0

            elif key.startswith("agent0_"):
                new_idx = int(key.split("_")[1])
                if new_idx != self.selected_agent0:
                    self.selected_agent0 = new_idx
                    self.selected_model0 = 0
                    self._model_scroll0 = 0

            elif key.startswith("agent1_"):
                new_idx = int(key.split("_")[1])
                if new_idx != self.selected_agent1:
                    self.selected_agent1 = new_idx
                    self.selected_model1 = 0
                    self._model_scroll1 = 0

            elif key.startswith("model0_"):
                a0_name = self.agent_choices[self.selected_agent0]
                self._handle_model_click(key, "model0", a0_name, env_name)

            elif key.startswith("model1_"):
                a1_name = self.agent_choices[self.selected_agent1]
                self._handle_model_click(key, "model1", a1_name, env_name)

            elif key == "start":
                if self._can_start():
                    self._start_game()

    # --- Decouverte de modeles ---

    @staticmethod
    def _agent_needs_model(agent_name: str) -> bool:
        return agent_name not in AGENTS_WITHOUT_MODELS

    def _scan_models(self, agent_name: str, env_name: str) -> list[tuple[str, str]]:
        """Retourne [(label, chemin_run_dir), ...] pour les modeles entraines disponibles."""
        agent_dir = os.path.join("results", env_name, agent_name)
        if not os.path.isdir(agent_dir):
            return []

        choices = []

        best_dir = os.path.join(agent_dir, "best")
        if os.path.isdir(best_dir) and (
            os.path.isfile(os.path.join(best_dir, "model.pt"))
            or glob.glob(os.path.join(glob.escape(best_dir), "model_*.pt"))
        ):
            choices.append(("Best", best_dir))

        runs = []
        for entry in sorted(os.listdir(agent_dir)):
            if entry == "best":
                continue
            run_dir = os.path.join(agent_dir, entry)
            if not os.path.isdir(run_dir):
                continue
            if not glob.glob(os.path.join(glob.escape(run_dir), "model_*.pt")):
                continue
            m = re.search(r"_seed(\d+)$", entry)
            seed = m.group(1) if m else "?"
            param_prefix = entry[: entry.rfind("_seed")] if m else entry
            runs.append((seed, param_prefix, run_dir))

        param_groups = sorted(set(r[1] for r in runs))
        multi_config = len(param_groups) > 1
        group_map = {p: i + 1 for i, p in enumerate(param_groups)}

        for seed, prefix, run_dir in runs:
            if multi_config:
                label = f"cfg{group_map[prefix]} s{seed}"
            else:
                label = f"seed {seed}"
            choices.append((label, run_dir))

        return choices

    def _get_model_choices(self, agent_name: str, env_name: str) -> list[tuple[str, str]]:
        key = (env_name, agent_name)
        if key not in self._cached_model_choices:
            self._cached_model_choices[key] = self._scan_models(agent_name, env_name)
        return self._cached_model_choices[key]

    @staticmethod
    def _find_model_in_dir(run_dir: str) -> str | None:
        """Trouve le fichier modele : model.pt (best/) ou le dernier model_{N}.pt."""
        model_pt = os.path.join(run_dir, "model.pt")
        if os.path.isfile(model_pt):
            return model_pt
        best_path = None
        best_cp = -1
        for path in glob.glob(os.path.join(glob.escape(run_dir), "model_*.pt")):
            fname = os.path.basename(path)
            try:
                cp = int(fname.replace("model_", "").replace(".pt", ""))
            except ValueError:
                continue
            if cp > best_cp:
                best_cp = cp
                best_path = path
        return best_path

    def _create_agent_for_play(self, agent_name: str, run_dir: str | None, env):
        """Instancie un agent, charge la config et le modele depuis un repertoire de resultats."""
        if run_dir is None:
            return get_agent(agent_name, env)

        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        agent_params = cfg.get("agent_params", {})

        agent = get_agent(agent_name, env, agent_params)

        model_path = self._find_model_in_dir(run_dir)
        if model_path:
            agent.load(model_path)
            print(f"[GUI] Loaded: {model_path}")

        return agent

    def _can_start(self) -> bool:
        env_name = self.env_choices[self.selected_env]
        a0_name = self.agent_choices[self.selected_agent0]
        if self._agent_needs_model(a0_name):
            if not self._get_model_choices(a0_name, env_name):
                return False
        test_env = get_env(env_name)
        if test_env.is_adversarial():
            a1_name = self.agent_choices[self.selected_agent1]
            if self._agent_needs_model(a1_name):
                if not self._get_model_choices(a1_name, env_name):
                    return False
        return True

    def _resolve_run_dir(self, agent_name: str, env_name: str,
                         selected_idx: int) -> str | None:
        """Obtient le run_dir pour un agent base sur un modele, ou None."""
        if not self._agent_needs_model(agent_name):
            return None
        choices = self._get_model_choices(agent_name, env_name)
        if not choices:
            return None
        idx = min(selected_idx, len(choices) - 1)
        return choices[idx][1]

    def _start_game(self):
        self.env_name = self.env_choices[self.selected_env]
        self.env = get_env(self.env_name)

        a0_name = self.agent_choices[self.selected_agent0]
        run_dir0 = self._resolve_run_dir(a0_name, self.env_name,
                                         self.selected_model0)
        agent0 = self._create_agent_for_play(a0_name, run_dir0, self.env)
        self.agent_names = [a0_name]

        if self.env.is_adversarial():
            a1_name = self.agent_choices[self.selected_agent1]
            run_dir1 = self._resolve_run_dir(a1_name, self.env_name,
                                             self.selected_model1)
            agent1 = self._create_agent_for_play(a1_name, run_dir1, self.env)
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

    # --- Logique de jeu ---

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
            # Le joueur qui vient d'agir a gagne ; mais current_player a deja change
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

    # --- Gestion des entrees ---

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
        """Associe les touches clavier aux actions pour le jeu humain LineWorld/GridWorld."""
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

    # --- Clic TicTacToe ---

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

    # --- Clic Bobail ---

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
            # Deuxieme clic : essayer de se deplacer vers cette case
            if clicked_idx in self.valid_moves_from_selected:
                action = self.valid_moves_from_selected[clicked_idx]
                self._do_step_human(action)
                return
            # Clic ailleurs : deselectionner et essayer de selectionner
            self.selected_piece = None
            self.valid_moves_from_selected = {}

        # Premier clic : selectionner une piece source
        moves_from_here = {}
        for a in available:
            from_cell = a // (BOBAIL_SIZE * BOBAIL_SIZE)
            to_cell = a % (BOBAIL_SIZE * BOBAIL_SIZE)
            if from_cell == clicked_idx:
                moves_from_here[to_cell] = a

        if moves_from_here:
            self.selected_piece = clicked_idx
            self.valid_moves_from_selected = moves_from_here

    # --- Dessin : Jeu ---

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

        # Bouton retour au menu
        self._back_btn = Button(
            pygame.Rect(panel_x + 20, WINDOW_H - 60, 170, 40),
            "Back to Menu", self.font_sm, color=DARK_GRAY)
        self._back_btn.draw(self.screen, mouse_pos)

    # --- LineWorld (Monde lineaire) ---

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

    # --- GridWorld (Monde en grille) ---

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

    # --- TicTacToe (Morpion) ---

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

                # Surligner les destinations de deplacement valides
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

        # Dessiner les indicateurs de rangee de camp
        for c in range(BOBAIL_SIZE):
            # Camp J1 = ligne 4 (bas), Camp J2 = ligne 0 (haut)
            x_top = board_rect.x + c * cell_w + cell_w // 2
            x_bot = x_top
            txt_top = self.font_sm.render("^", True, (RED[0], RED[1], RED[2], 80))
            txt_bot = self.font_sm.render("v", True, (ACCENT[0], ACCENT[1], ACCENT[2], 80))
            self.screen.blit(txt_top, txt_top.get_rect(center=(x_top, board_rect.y - 12)))
            self.screen.blit(txt_bot, txt_bot.get_rect(center=(x_bot, board_rect.y + board_rect.height + 12)))
