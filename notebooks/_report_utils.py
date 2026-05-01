"""Shared helpers for the analysis notebooks.

Loaders unify the four schemas found in `results/`:
- single-player metrics.csv (line_world / grid_world)
- adversarial metrics_reeval.csv (combined + p0/p1 columns)
- single-player metrics_reeval.csv (line_world / grid_world)
- planning metrics.csv (budget instead of checkpoint, with win/draw/loss)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ENV_LIST = ["line_world", "grid_world", "tictactoe", "bobail"]
ADVERSARIAL_ENVS = {"tictactoe", "bobail"}

LEARNING_AGENTS = [
    "random",
    "tabular_q",
    "dqn",
    "ddqn",
    "ddqn_er",
    "ddqn_per",
    "reinforce",
    "reinforce_mean_baseline",
    "reinforce_critic",
    "ppo",
]
PLANNING_AGENTS = ["random_rollout", "mcts"]
ALL_AGENTS = LEARNING_AGENTS + PLANNING_AGENTS

AGENT_FAMILY = {
    "random": "baseline",
    "tabular_q": "tabular",
    "dqn": "value-based",
    "ddqn": "value-based",
    "ddqn_er": "value-based",
    "ddqn_per": "value-based",
    "reinforce": "policy-gradient",
    "reinforce_mean_baseline": "policy-gradient",
    "reinforce_critic": "policy-gradient",
    "ppo": "policy-gradient",
    "random_rollout": "planning",
    "mcts": "planning",
}

FAMILY_COLOR = {
    "baseline": "#9E9E9E",
    "tabular": "#795548",
    "value-based": "#1976D2",
    "policy-gradient": "#E91E63",
    "planning": "#388E3C",
}

CANONICAL_CHECKPOINTS = [1000, 10000, 50000, 100000, 1000000]


def project_root() -> Path:
    p = Path.cwd()
    if p.name == "notebooks":
        p = p.parent
    return p


def results_dir() -> Path:
    return project_root() / "results"


def discover_runs(env_name: str):
    """Return (learning_runs, planning_runs).

    Each run is a dict {agent, run_dir, run_name, seed, config}.
    Skips 'best/' (curated copy, would duplicate data).
    """
    learning, planning = [], []
    env_path = results_dir() / env_name
    if not env_path.exists():
        return learning, planning

    for agent_dir in sorted(env_path.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent_name = agent_dir.name
        for run_dir in sorted(agent_dir.iterdir()):
            if not run_dir.is_dir() or run_dir.name == "best":
                continue
            cfg_path = run_dir / "config.yaml"
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            entry = {
                "env": env_name,
                "agent": agent_name,
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "seed": cfg.get("seed", ""),
                "config": cfg,
            }
            if run_dir.name.startswith("budget_sweep") or agent_name in PLANNING_AGENTS:
                planning.append(entry)
            else:
                learning.append(entry)
    return learning, planning


def _normalize_adversarial(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "mean_reward_combined": "mean_reward",
        "mean_steps_combined": "mean_steps",
        "win_rate_combined": "win_rate",
        "draw_rate_combined": "draw_rate",
        "loss_rate_combined": "loss_rate",
        "termination_rate_combined": "termination_rate",
        "truncation_rate_combined": "truncation_rate",
    }
    for src, dst in rename.items():
        if src in df.columns:
            df[dst] = df[src]
    return df


def load_eval_metrics(run_entry: dict) -> pd.DataFrame:
    """Return a normalized eval frame for a learning run.

    Tries metrics_reeval.csv first, falls back to metrics.csv.
    Always exposes a unified column set with NaN for missing fields.
    """
    run_dir = run_entry["run_dir"]
    reeval = run_dir / "metrics_reeval.csv"
    fallback = run_dir / "metrics.csv"

    df = None
    if reeval.exists() and reeval.stat().st_size > 0:
        df = pd.read_csv(reeval)
        if df.empty:
            df = None
    if df is None and fallback.exists():
        df = pd.read_csv(fallback)
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    is_adv = "mean_reward_combined" in df.columns
    if is_adv:
        df = _normalize_adversarial(df)
    for col in [
        "win_rate", "draw_rate", "loss_rate",
        "win_rate_p0", "win_rate_p1",
        "mean_reward_p0", "mean_reward_p1",
        "termination_rate", "truncation_rate",
    ]:
        if col not in df.columns:
            df[col] = np.nan

    df["env"] = run_entry["env"]
    df["agent"] = run_entry["agent"]
    df["run_name"] = run_entry["run_name"]
    df["seed"] = run_entry["seed"]
    return df


def load_training_curve(run_entry: dict) -> pd.DataFrame:
    p = run_entry["run_dir"] / "training_curve.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["env"] = run_entry["env"]
    df["agent"] = run_entry["agent"]
    df["run_name"] = run_entry["run_name"]
    df["seed"] = run_entry["seed"]
    return df


def load_planning_metrics(run_entry: dict) -> pd.DataFrame:
    p = run_entry["run_dir"] / "metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["env"] = run_entry["env"]
    df["agent"] = run_entry["agent"]
    df["run_name"] = run_entry["run_name"]
    df["seed"] = run_entry["seed"]
    return df


def load_all_eval(envs=ENV_LIST) -> pd.DataFrame:
    """Concat normalized eval frames across all envs (learning runs only)."""
    frames = []
    for env in envs:
        learning, _ = discover_runs(env)
        for r in learning:
            f = load_eval_metrics(r)
            if not f.empty:
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_planning(envs=ENV_LIST) -> pd.DataFrame:
    frames = []
    for env in envs:
        _, planning = discover_runs(env)
        for r in planning:
            f = load_planning_metrics(r)
            if not f.empty:
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_all_training(envs=ENV_LIST) -> pd.DataFrame:
    frames = []
    for env in envs:
        learning, _ = discover_runs(env)
        for r in learning:
            f = load_training_curve(r)
            if not f.empty:
                frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_run_per_agent(df_eval: pd.DataFrame, env: str) -> pd.DataFrame:
    """For each agent on a given env, return the row with the highest mean_reward
    across all (run_name, checkpoint) combinations. Tie-break on lower mean_steps.
    """
    sub = df_eval[df_eval["env"] == env].copy()
    if sub.empty:
        return sub
    sub = sub.dropna(subset=["mean_reward"])
    if sub.empty:
        return sub
    sub = sub.sort_values(["mean_reward", "mean_steps"], ascending=[False, True])
    return sub.groupby("agent", as_index=False).head(1).reset_index(drop=True)


def best_seed_run(df_eval: pd.DataFrame, env: str, agent: str) -> str | None:
    """Return the run_name (string) of the best-performing seed for (env, agent)."""
    sub = df_eval[(df_eval["env"] == env) & (df_eval["agent"] == agent)].copy()
    if sub.empty:
        return None
    sub = sub.dropna(subset=["mean_reward"])
    if sub.empty:
        return None
    best_per_run = (
        sub.groupby("run_name", as_index=False)["mean_reward"].max()
        .sort_values("mean_reward", ascending=False)
    )
    return best_per_run.iloc[0]["run_name"]


def aggregate_at_checkpoints(
    df_eval: pd.DataFrame,
    env: str,
    checkpoints: list[int] | None = None,
) -> pd.DataFrame:
    """Return one row per agent with mean_reward / mean_steps / mean_action_time_ms
    at each requested checkpoint (averaged over seeds, on the best-hyperparam run)."""
    if checkpoints is None:
        checkpoints = CANONICAL_CHECKPOINTS
    sub = df_eval[df_eval["env"] == env].copy()
    if sub.empty:
        return sub

    rows = []
    for agent in sub["agent"].unique():
        agent_sub = sub[sub["agent"] == agent]
        run_name = best_seed_run(df_eval, env, agent)
        if run_name is None:
            continue
        agent_runs = sub[(sub["agent"] == agent) & (sub["run_name"].apply(
            lambda r: r.split("_seed")[0] == run_name.split("_seed")[0]
        ))]
        for ckpt in checkpoints:
            ckpt_rows = agent_runs[agent_runs["checkpoint"] == ckpt]
            if ckpt_rows.empty:
                continue
            rows.append({
                "agent": agent,
                "checkpoint": ckpt,
                "mean_reward": ckpt_rows["mean_reward"].mean(),
                "mean_steps": ckpt_rows["mean_steps"].mean(),
                "mean_action_time_ms": ckpt_rows["mean_action_time_ms"].mean(),
                "win_rate": ckpt_rows["win_rate"].mean(),
                "draw_rate": ckpt_rows["draw_rate"].mean(),
                "loss_rate": ckpt_rows["loss_rate"].mean(),
                "termination_rate": ckpt_rows["termination_rate"].mean(),
                "n_seeds": len(ckpt_rows),
            })
    return pd.DataFrame(rows)


def setup_plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def agent_color(agent: str):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    idx = ALL_AGENTS.index(agent) if agent in ALL_AGENTS else 0
    return cmap(idx % 20)


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter analysis layer
#
# Source of truth: `config.yaml` inside each run dir under results/<env>/<agent>/.
# Run-dir names are lossy (omit training.num_episodes, eval.num_games, opponent,
# and use inconsistent abbreviations across families) — never parse them.
# ─────────────────────────────────────────────────────────────────────────────


def flatten_config(cfg: dict) -> dict:
    """YAML run config → flat dict, prefixed by section.

    agent_params.* → hp_*       (lr, gamma, hidden_layers, ...)
    training.*     → train_*    (num_episodes, max_steps_per_episode)
    eval.*         → eval_*     (num_games, checkpoints)
    top-level `budgets` (planning) → eval_budgets

    Lists are converted to tuples so the column is hashable (groupby-friendly).
    Missing fields stay missing — we don't impute defaults, so absence is signal.
    """
    out: dict = {}
    if not cfg:
        return out

    for key in ("env", "agent", "opponent", "seed"):
        if key in cfg:
            out[key] = cfg[key]

    def _hashable(v):
        return tuple(v) if isinstance(v, list) else v

    for k, v in (cfg.get("agent_params") or {}).items():
        out[f"hp_{k}"] = _hashable(v)
    for k, v in (cfg.get("training") or {}).items():
        out[f"train_{k}"] = _hashable(v)
    for k, v in (cfg.get("eval") or {}).items():
        out[f"eval_{k}"] = _hashable(v)

    if "budgets" in cfg:
        out["eval_budgets"] = _hashable(cfg["budgets"])

    return out


def load_runs_index(envs=ENV_LIST) -> pd.DataFrame:
    """One row per run dir (across all envs), with HPs flattened from config.yaml.

    Columns include: env, agent, family, kind ('learning'|'planning'),
    run_name, run_dir, seed, opponent, plus all hp_* / train_* / eval_* keys
    that appeared in any config (NaN where a run didn't set that key).
    """
    rows = []
    for env in envs:
        learning, planning = discover_runs(env)
        for r in learning:
            flat = flatten_config(r["config"])
            flat["run_name"] = r["run_name"]
            flat["run_dir"] = str(r["run_dir"])
            flat["family"] = AGENT_FAMILY.get(flat.get("agent"), "unknown")
            flat["kind"] = "learning"
            rows.append(flat)
        for r in planning:
            flat = flatten_config(r["config"])
            flat["run_name"] = r["run_name"]
            flat["run_dir"] = str(r["run_dir"])
            flat["family"] = AGENT_FAMILY.get(flat.get("agent"), "unknown")
            flat["kind"] = "planning"
            rows.append(flat)
    return pd.DataFrame(rows)


def load_runs_long(envs=ENV_LIST) -> pd.DataFrame:
    """One row per (run, eval-axis-value), enriched with all HP columns.

    eval_axis_kind = 'checkpoint' (learning runs) | 'budget' (planning runs).
    eval_axis_value holds the value (training episodes count, or sim budget).

    Metric columns are the union from learning and planning loaders:
    mean_reward, mean_steps, mean_action_time_ms,
    win_rate, draw_rate, loss_rate, termination_rate, truncation_rate,
    plus adversarial-only win_rate_p0/p1, mean_reward_p0/p1.

    Any HP that varies across runs becomes a groupby axis.
    """
    idx = load_runs_index(envs)

    pieces = []
    for env in envs:
        learning, planning = discover_runs(env)
        for r in learning:
            df = load_eval_metrics(r)
            if df.empty:
                continue
            df = df.rename(columns={"checkpoint": "eval_axis_value"})
            df["eval_axis_kind"] = "checkpoint"
            df["run_dir"] = str(r["run_dir"])
            pieces.append(df)
        for r in planning:
            df = load_planning_metrics(r)
            if df.empty:
                continue
            df = df.rename(columns={"budget": "eval_axis_value"})
            df["eval_axis_kind"] = "budget"
            df["run_dir"] = str(r["run_dir"])
            pieces.append(df)

    if not pieces:
        return pd.DataFrame()

    long_df = pd.concat(pieces, ignore_index=True)
    # `run_name` is NOT unique across envs (e.g. 'budget_sweep_seed42' exists
    # under every env × planning agent), so we merge on `run_dir` which is.
    # Drop loader-injected cols (we'll get them from idx).
    long_df = long_df.drop(
        columns=[c for c in ("env", "agent", "seed", "run_name") if c in long_df.columns]
    )
    return long_df.merge(idx, on="run_dir", how="left")


def varying_hparams(
    df: pd.DataFrame,
    prefixes: tuple[str, ...] = ("hp_", "train_", "eval_"),
) -> list[str]:
    """Return columns with given prefixes that take >1 distinct value in `df`.

    Useful for spotting which knobs were actually swept in a slice
    (e.g. df.query("env == 'bobail' and agent == 'ppo'")).
    NaN counts as its own value, so a column where some runs set it and
    others don't will surface as varying — which is the right signal.
    """
    cols = [c for c in df.columns if any(c.startswith(p) for p in prefixes)]
    out = []
    for c in cols:
        try:
            n = df[c].nunique(dropna=False)
        except TypeError:
            # Defensive: unhashable values (shouldn't happen — we tuple-ify lists).
            n = df[c].apply(repr).nunique()
        if n > 1:
            out.append(c)
    return out


def hparam_summary(
    df_long: pd.DataFrame,
    env: str,
    agent: str,
    metric: str = "mean_reward",
    eval_axis_value: int | None = None,
) -> pd.DataFrame:
    """One row per (varying HP combination × seed) for (env, agent), at a given
    eval-axis value (default: the last available checkpoint/budget per run).

    Columns: the varying HPs of that slice + seed + eval_axis_value + metric(s).
    Constant HPs are omitted to keep the table compact.

    This is the canonical "show me the effect of changing X on Y" table.
    """
    sub = df_long[(df_long["env"] == env) & (df_long["agent"] == agent)].copy()
    if sub.empty:
        return sub

    if eval_axis_value is None:
        sub = sub.sort_values("eval_axis_value").groupby("run_name", as_index=False).tail(1)
    else:
        sub = sub[sub["eval_axis_value"] == eval_axis_value]
    if sub.empty:
        return sub

    var_cols = varying_hparams(sub)
    keep = var_cols + ["seed", "eval_axis_value", metric]
    extras = [c for c in ("mean_steps", "mean_action_time_ms", "win_rate") if c in sub.columns and c != metric]
    keep += extras
    keep = [c for c in keep if c in sub.columns]
    return sub[keep].sort_values(metric, ascending=False).reset_index(drop=True)


def config_signature_series(df: pd.DataFrame) -> pd.Series:
    """Per-row signature string from hp_* and train_* columns (for grouping configs)."""
    return _config_fingerprint_series(df)


def _config_fingerprint_series(df: pd.DataFrame) -> pd.Series:
    cols = sorted(c for c in df.columns if c.startswith(("hp_", "train_")))
    parts_list = []
    for _, row in df.iterrows():
        parts = []
        for c in cols:
            v = row.get(c)
            if pd.isna(v):
                continue
            parts.append(f"{c}={v}")
        parts_list.append("|".join(parts) if parts else "(defaults)")
    return pd.Series(parts_list, index=df.index)


def compact_config_table(idx: pd.DataFrame) -> pd.DataFrame:
    """One row per (env, agent, config fingerprint): n_runs and truncated signature."""
    if idx.empty:
        return pd.DataFrame(columns=["env", "agent", "n_runs", "config_signature"])
    work = idx.copy()
    work["config_signature"] = _config_fingerprint_series(work)
    rows = []
    for (env, agent), g in work.groupby(["env", "agent"], sort=True):
        for sig, gg in g.groupby("config_signature", sort=True):
            rows.append({
                "env": env,
                "agent": agent,
                "n_runs": len(gg),
                "config_signature": sig if len(sig) <= 300 else sig[:297] + "...",
            })
    return pd.DataFrame(rows).sort_values(["env", "agent", "n_runs"], ascending=[True, True, False]).reset_index(drop=True)


def plot_hp_coverage(idx: pd.DataFrame, ax=None, figsize=(12, 6)):
    """Heatmap: agents × HP columns, cell = number of distinct values (incl. NaN as level)."""
    import matplotlib.pyplot as plt

    if idx.empty:
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No runs", ha="center", va="center")
        return ax

    hp_cols = sorted(
        c for c in idx.columns
        if c.startswith("hp_") and idx[c].notna().any()
    )
    agents = sorted(idx["agent"].dropna().unique())
    if not hp_cols or not agents:
        if ax is None:
            _, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No HP columns", ha="center", va="center")
        return ax

    mat = np.zeros((len(agents), len(hp_cols)))
    for i, ag in enumerate(agents):
        sub = idx[idx["agent"] == ag]
        for j, col in enumerate(hp_cols):
            mat[i, j] = sub[col].nunique(dropna=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0)
    ax.set_xticks(range(len(hp_cols)))
    ax.set_xticklabels(hp_cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels(agents, fontsize=8)
    ax.set_title("Couverture HPs : nombre de valeurs distinctes par agent")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=7, color="white" if v > 1 else "black")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="nb valeurs")
    return ax


def plot_experiment_tree(idx: pd.DataFrame, ax=None, figsize=None):
    """Horizontal tree: env → famille → agent → configs (fingerprints), with n_runs per leaf."""
    import matplotlib.pyplot as plt

    if idx.empty:
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 2))
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No runs", ha="center", va="center", transform=ax.transAxes)
        return ax

    work = idx.copy()
    work["config_signature"] = _config_fingerprint_series(work)
    leaves_df = (
        work.groupby(["env", "family", "agent", "config_signature"], sort=True)
        .size()
        .reset_index(name="n_runs")
    )
    leaves_df = leaves_df.sort_values(
        ["env", "family", "agent", "config_signature"]
    ).reset_index(drop=True)
    n = len(leaves_df)
    if figsize is None:
        figsize = (18, max(6, 0.22 * n))

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    leaf_y = np.arange(n, dtype=float)
    env_y: dict = {}
    for e, g in leaves_df.groupby("env", sort=False):
        env_y[e] = float(g.index.mean())
    family_y: dict = {}
    for (e, fam), g in leaves_df.groupby(["env", "family"], sort=False):
        family_y[(e, fam)] = float(g.index.mean())
    agent_y: dict = {}
    for (e, fam, ag), g in leaves_df.groupby(["env", "family", "agent"], sort=False):
        agent_y[(e, fam, ag)] = float(g.index.mean())

    xs = (0.0, 1.0, 2.0, 3.0)
    ax.set_xlim(-0.35, 4.2)
    ax.set_ylim(-0.5, max(n - 0.5, 0.5))
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title("Arbre des expérimentations (feuille = signature config × seeds)")

    for e, y in env_y.items():
        ax.scatter([xs[0]], [y], s=140, c=FAMILY_COLOR.get("baseline", "#9E9E9E"), zorder=2, edgecolors="black", linewidths=0.5)
        ax.text(xs[0] - 0.12, y, str(e), ha="right", va="center", fontsize=9, fontweight="bold")

    for (e, fam), y in family_y.items():
        ax.scatter([xs[1]], [y], s=120, c=FAMILY_COLOR.get(fam, "#9E9E9E"), zorder=2, edgecolors="black", linewidths=0.5)
        ax.text(xs[1] - 0.1, y, fam, ha="right", va="center", fontsize=8)
        ax.plot([xs[0], xs[1]], [env_y[e], y], color="#90A4AE", linewidth=1.0, zorder=0)

    seen_ag = set()
    for (e, fam, ag), y in agent_y.items():
        key = (e, fam, ag)
        if key in seen_ag:
            continue
        seen_ag.add(key)
        ax.scatter([xs[2]], [y], s=100, c=FAMILY_COLOR.get(fam, "#9E9E9E"), zorder=2, edgecolors="black", linewidths=0.5)
        ax.text(xs[2] - 0.08, y, ag, ha="right", va="center", fontsize=7)
        ax.plot([xs[1], xs[2]], [family_y[(e, fam)], y], color="#B0BEC5", linewidth=0.9, zorder=0)

    for i, row in leaves_df.iterrows():
        e, fam, ag = row["env"], row["family"], row["agent"]
        sig = row["config_signature"]
        if len(sig) > 85:
            sig = sig[:82] + "..."
        label = f"{int(row['n_runs'])}× | {sig}"
        yi = leaf_y[i]
        ax.plot([xs[2], xs[3]], [agent_y[(e, fam, ag)], yi], color="#CFD8DC", linewidth=0.7, zorder=0)
        ax.text(xs[3] + 0.02, yi, label, va="center", fontsize=6, clip_on=False)

    return ax
