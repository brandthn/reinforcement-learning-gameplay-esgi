"""One-off: rebuild hyperparams_methodology.ipynb. Run from repo root: python notebooks/_rebuild_hyperparams_nb.py"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parent / "hyperparams_methodology.ipynb"


def cell(typ, src, cid=None):
    c = {"cell_type": typ, "metadata": {}, "source": src if isinstance(src, list) else [src]}
    if cid:
        c["id"] = cid
    if typ == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c


cells = [
    cell("markdown", """# Analyse hyperparamètres — méthodologie & effet par famille

Pour chaque famille d'algorithmes, effet observé des hyperparamètres-clés sur les performances. Source de vérité : `config.yaml` de chaque run sous `results/` (le nom du dossier est incomplet et n'est pas parsé).

**Plan**

0. Méthodologie de choix des hyperparamètres  
1. Setup & chargement des runs  
2. Catalogue visuel des expérimentations (arbre, couverture HPs, table compacte)  
3. Inventaire des runs  
4. Famille tabular & value-based  
5. Famille policy-gradient  
6. Famille planning  
7. Synthèse multi-environnements & implications (soutenance)"""),
    cell("markdown", """## 0. Méthodologie de choix des hyperparamètres

- **Point de départ** : ordres de grandeur usuels (Q-learning / DQN : `gamma` proche de 1, `lr` modéré ; PPO : clipping 0.2, GAE, etc.).
- **Configs reproductibles** : un fichier YAML dans `configs_done/` décrit une expérience ; les valeurs réellement utilisées sont celles du `config.yaml` dans chaque dossier de run sous `results/`.
- **Sweeps** : lorsqu'un paramètre méritait d'être exploré, plusieurs YAML ou plusieurs seeds ont été lancés ; la recherche reste **manuelle** (pas d'optimisation bayésienne).
- **Seeds** : 2–3 seeds quand le coût CPU le permettait, 1 sinon.
- **Sélection « meilleure config »** : pour un couple (environnement, agent), on retient le `run_dir` qui maximise `mean_reward` au **dernier** point d'évaluation disponible (dernier checkpoint ou dernier budget), avec préférence pour des `mean_steps` plus faibles en cas d'égalité sur le même checkpoint.
- **Limites** : peu de seeds → intervalles de confiance larges ; comparaisons « meilleure config par variante » **mélangent** l'algorithme et l'effort de réglage (voir caveats dans les ablations)."""),
    cell("markdown", "## 1. Setup"),
    cell("code", """import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path.cwd().parent / "notebooks"))

from _report_utils import (
    ENV_LIST,
    ADVERSARIAL_ENVS,
    AGENT_FAMILY,
    FAMILY_COLOR,
    load_runs_index,
    load_runs_long,
    varying_hparams,
    hparam_summary,
    setup_plot_style,
    config_signature_series,
    compact_config_table,
    plot_experiment_tree,
    plot_hp_coverage,
)

setup_plot_style()
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)"""),
    cell("code", """idx = load_runs_index()
dfl = load_runs_long()
print(f"Index : {len(idx)} runs | Long frame : {len(dfl)} (run, eval-axis) rows")"""),
    cell("markdown", """## 2. Catalogue visuel des expérimentations

- **Arbre** : environnement → famille → agent → signature de config (hyperparamètres + training) avec le nombre de runs (souvent seeds).
- **Heatmap** : pour chaque agent, nombre de **valeurs distinctes** observées par colonne `hp_*` (0 = non utilisé).
- **Table compacte** : une ligne par triplet (env, agent, signature), colonne `n_runs`."""),
    cell("code", """fig, ax = plt.subplots(figsize=(18, max(6, 0.18 * len(idx))))
plot_experiment_tree(idx, ax=ax)
plt.tight_layout()"""),
    cell("code", """fig, ax = plt.subplots(figsize=(14, 5))
plot_hp_coverage(idx, ax=ax)
plt.tight_layout()"""),
    cell("code", """cc = compact_config_table(idx)
display(cc)"""),
    cell("markdown", """Helpers locaux :

- `plot_hp_curves` — utile quand **un seul** HP varie à la fois ; sinon les courbes sont **confondues** avec d'autres changements (voir sections DDQN / PPO).
- `bar_at_last` — barres au dernier `eval_axis_value`.
- `plot_runs_curves` — une courbe par `run_dir` (toute la config agrégée)."""),
    cell("code", """def plot_hp_curves(df, hp_col, metric="mean_reward", ax=None, title=None, log_x=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    g = (df.groupby([hp_col, "eval_axis_value"])[metric]
           .agg(["mean", "std", "count"]).reset_index())
    for val, sub in g.groupby(hp_col):
        sub = sub.sort_values("eval_axis_value")
        line, = ax.plot(sub["eval_axis_value"], sub["mean"], marker="o", label=f"{hp_col}={val}")
        ax.fill_between(sub["eval_axis_value"],
                        sub["mean"] - sub["std"].fillna(0),
                        sub["mean"] + sub["std"].fillna(0),
                        color=line.get_color(), alpha=0.15)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("eval_axis_value (épisodes ou budget)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"Effet de {hp_col} sur {metric}")
    ax.legend(fontsize=8, loc="best")
    return ax


def bar_at_last(df, group_col, metric="mean_reward", ax=None, title=None, color="#1976D2"):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    last = df.sort_values("eval_axis_value").groupby("run_dir").tail(1).copy()
    if isinstance(group_col, (list, tuple)):
        last["_label"] = last[list(group_col)].astype(str).agg(" / ".join, axis=1)
        group_col = "_label"
    g = last.groupby(group_col)[metric].agg(["mean", "std"]).sort_values("mean")
    ax.barh(g.index.astype(str), g["mean"], xerr=g["std"].fillna(0),
            color=color, alpha=0.85, edgecolor="black")
    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} (dernier ckpt) par {group_col}")
    return ax


def plot_runs_curves(df, metric="mean_reward", ax=None, title=None, log_x=True, max_label=48):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    for run_dir, sub in df.groupby("run_dir"):
        sub = sub.sort_values("eval_axis_value")
        label = str(run_dir).split("/")[-1][:max_label]
        ax.plot(sub["eval_axis_value"], sub[metric], marker="o", markersize=3, label=label, alpha=0.85)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("eval_axis_value")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} par run_dir (config complète)")
    ax.legend(fontsize=5, loc="best", ncol=2)
    return ax"""),
    cell("markdown", "## 3. Inventaire des runs"),
    cell("code", """inv = idx.groupby(["env", "family", "agent"]).size().unstack(fill_value=0)
display(inv)

runs_per_ea = idx.groupby(["env", "agent"]).size().reset_index(name="n_runs")
cc = compact_config_table(idx)
n_cfg = cc.groupby(["env", "agent"]).size().reset_index(name="n_distinct_configs")
inv2 = runs_per_ea.merge(n_cfg, on=["env", "agent"], how="left").sort_values(["env", "agent"])
display(inv2)"""),
    cell("code", """multi_cfg = (idx.groupby(["env", "agent"]).size().reset_index(name="n_runs")
                .query("n_runs > 1").sort_values("n_runs", ascending=False))
multi_cfg"""),
    cell("markdown", """Bobail concentre la majorité des ablations. PPO/Bobail (15 runs) et PPO/TicTacToe (13 runs) sont parmi les sweeps les plus larges ; les `reinforce*` ont souvent 2 runs = 2 seeds sur une même config YAML."""),
    cell("markdown", """## 4. Famille tabular & value-based

- **tabular_q** : Q-learning tabulaire.
- **dqn** : réseau + replay + réseau cible.
- **ddqn** : double Q pour limiter la sur-estimation.
- **ddqn_er** : même famille avec paramètres de replay explicites (`learning_starts`, etc.).
- **ddqn_per** : replay prioritaire (`per_alpha`, annealing `per_beta_*`)."""),
    cell("markdown", """### 4.1 Ablation DQN → DDQN → +ER → +PER (Bobail, dernier checkpoint, meilleure config par agent)

**Caveat** : chaque colonne est la **meilleure** `run_dir` trouvée pour cette variante. L'écart inclut donc à la fois l'algorithme et l'effort de réglage (nombre de YAML, seeds). ER/PER ajoutent notamment `learning_starts`, absent des runs DQN/DDQN listés ici."""),
    cell("code", """agents = ["dqn", "ddqn", "ddqn_er", "ddqn_per"]
sub = dfl.query("env == 'bobail' and agent in @agents")

last = sub.sort_values("eval_axis_value").groupby("run_dir").tail(1)
best = (last.groupby(["agent", "run_dir"], as_index=False)["mean_reward"].mean()
            .sort_values(["agent", "mean_reward"], ascending=[True, False])
            .groupby("agent", as_index=False).head(1))
present = [a for a in agents if a in best["agent"].values]
best_ordered = best.set_index("agent").loc[present].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax0, ax1 = axes
ax0.barh(best_ordered["agent"], best_ordered["mean_reward"],
         color=FAMILY_COLOR["value-based"], alpha=0.85, edgecolor="black")
ax0.set_xlabel("mean_reward (Bobail, dernier ckpt, meilleure config)")
ax0.set_title("Ablation DQN → DDQN → +ER → +PER")
ax0.axvline(0, color="black", linewidth=0.5)

rng = np.random.default_rng(42)
for j, ag in enumerate(agents):
    pts = last[last["agent"] == ag]["mean_reward"].values
    if len(pts) == 0:
        continue
    x = rng.normal(j, 0.06, size=len(pts))
    ax1.scatter(x, pts, alpha=0.55, s=36, c="#78909C", edgecolors="none")
    row = best_ordered[best_ordered["agent"] == ag]
    if not row.empty:
        ax1.scatter([j], [row["mean_reward"].values[0]], s=140, marker="*",
                    color=FAMILY_COLOR["value-based"], zorder=3, edgecolors="black")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_xticks(range(len(agents)))
ax1.set_xticklabels(agents)
ax1.set_ylabel("mean_reward")
ax1.set_title("Dispersion toutes configs×seeds (∗ = meilleure run_dir)")
plt.tight_layout()
display(best_ordered[["agent", "mean_reward", "run_dir"]])"""),
    cell("markdown", """Les quatre variantes deep value-based restent en moyenne **négatives** sur Bobail au dernier checkpoint disponible pour chaque run ; DDQN+ER (`lr=1e-4`, épisodes courts) domine légèrement ce sous-ensemble. La tabulation bat nettement cette famille sur le même opposant (section 4.3)."""),
    cell("markdown", """### 4.2 DDQN / Bobail — configs réelles (HPs non indépendants)

Les YAML « basse learning rate » changent en bloc `lr`, `target_update_freq`, `epsilon_decay_steps`, `max_steps_per_episode`, etc. **Ne pas** interpréter un graphe « univarié » `lr` seul comme causal."""),
    cell("code", """ddqn = dfl.query("env == 'bobail' and agent == 'ddqn'")
print("HPs qui varient sur cette slice :")
print(varying_hparams(ddqn))
print()
display(hparam_summary(dfl, "bobail", "ddqn"))"""),
    cell("code", """fig, ax = plt.subplots(figsize=(11, 5))
plot_runs_curves(ddqn, ax=ax, title="DDQN/Bobail — mean_reward par run_dir (toute la config)")
plt.tight_layout()"""),
    cell("markdown", """Les courbes par `run_dir` séparent deux régimes complets (YAML différents) plutôt qu'un effet isolé du learning rate."""),
    cell("markdown", """### 4.3 Tabular Q / Bobail — epsilon-decay et budget d'épisodes"""),
    cell("code", """tq = dfl.query("env == 'bobail' and agent == 'tabular_q'")
print("HPs qui varient :", varying_hparams(tq))
hparam_summary(dfl, "bobail", "tabular_q")"""),
    cell("code", """fig, axes = plt.subplots(1, 2, figsize=(14, 4))
if "hp_epsilon_decay_steps" in varying_hparams(tq):
    plot_hp_curves(tq, "hp_epsilon_decay_steps", "mean_reward", ax=axes[0],
                   title="Tabular Q / Bobail — epsilon decay")
if "train_num_episodes" in varying_hparams(tq):
    plot_hp_curves(tq, "train_num_episodes", "mean_reward", ax=axes[1],
                   title="Tabular Q / Bobail — budget d'épisodes")
plt.tight_layout()"""),
    cell("markdown", """Tabular Q atteint une récompense moyenne **positive** sur Bobail avec des épisodes courts (~11 steps) ; les variantes DQN analysées ici produisent des épisodes beaucoup plus longs (souvent proches du plafond `max_steps`), ce qui reflète une politique qui n'aligne pas le jeu dans ce réglage."""),
    cell("markdown", """## 5. Famille policy-gradient

- **REINFORCE** : policy gradient Monte Carlo.
- **+ mean baseline** : centrage des retours.
- **+ critic** : baseline apprise.
- **PPO** : critic + clipping + GAE (style A2C du projet)."""),
    cell("markdown", """### 5.1 Ablation REINFORCE → +mean → +critic → PPO (Bobail, dernier ckpt, meilleure config)

**Même caveat** que pour la famille value-based : meilleure `run_dir` par agent, donc algorithme + tuning."""),
    cell("code", """pg_agents = ["reinforce", "reinforce_mean_baseline", "reinforce_critic", "ppo"]
sub = dfl.query("env == 'bobail' and agent in @pg_agents")

last = sub.sort_values("eval_axis_value").groupby("run_dir").tail(1)
best = (last.groupby(["agent", "run_dir"], as_index=False)["mean_reward"].mean()
            .sort_values(["agent", "mean_reward"], ascending=[True, False])
            .groupby("agent", as_index=False).head(1))
present = [a for a in pg_agents if a in best["agent"].values]
best_ordered = best.set_index("agent").loc[present].reset_index()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
ax0, ax1 = axes
ax0.barh(best_ordered["agent"], best_ordered["mean_reward"],
         color=FAMILY_COLOR["policy-gradient"], alpha=0.85, edgecolor="black")
ax0.set_xlabel("mean_reward (Bobail, dernier ckpt, meilleure config)")
ax0.set_title("REINFORCE → +mean → +critic → PPO")
ax0.axvline(0, color="black", linewidth=0.5)

rng = np.random.default_rng(43)
for j, ag in enumerate(pg_agents):
    pts = last[last["agent"] == ag]["mean_reward"].values
    if len(pts) == 0:
        continue
    x = rng.normal(j, 0.06, size=len(pts))
    ax1.scatter(x, pts, alpha=0.55, s=36, c="#AD1457", edgecolors="none")
    row = best_ordered[best_ordered["agent"] == ag]
    if not row.empty:
        ax1.scatter([j], [row["mean_reward"].values[0]], s=140, marker="*",
                    color=FAMILY_COLOR["policy-gradient"], zorder=3, edgecolors="black")
ax1.axhline(0, color="black", linewidth=0.5)
ax1.set_xticks(range(len(pg_agents)))
ax1.set_xticklabels(pg_agents, rotation=15, ha="right")
ax1.set_ylabel("mean_reward")
ax1.set_title("Dispersion (∗ = meilleure run_dir)")
plt.tight_layout()
display(best_ordered[["agent", "mean_reward", "run_dir"]])"""),
    cell("code", """last_pg = sub.sort_values("eval_axis_value").groupby("run_dir").tail(1)
pg_summary = last_pg.groupby("agent")["mean_reward"].agg(["max", "mean", "std", "count"]).round(4)
display(pg_summary)"""),
    cell("markdown", """Sur **Bobail**, les policy gradients atteignent des scores élevés au dernier checkpoint ; le critic appris peut être légèrement derrière une simple moyenne de baseline sur ce sous-ensemble. Sur **TicTacToe**, les métriques PPO restent modérées (~0.42–0.47 au dernier checkpoint agrégé dans les tableaux du rapport) : la phrase « PPO au plafond » ne s'étend pas à tous les environnements."""),
    cell("markdown", """### 5.2 Sweep PPO / Bobail (15 runs)

Table **marginale** : moyenne au dernier checkpoint par valeur d'un HP (les autres HPs varient en même temps — lecture descriptive seulement)."""),
    cell("code", """ppo_b = dfl.query("env == 'bobail' and agent == 'ppo'")
print("HPs qui varient :", varying_hparams(ppo_b))
print()
display(hparam_summary(dfl, "bobail", "ppo").head(25))"""),
    cell("code", """last_p = ppo_b.sort_values("eval_axis_value").groupby("run_dir").tail(1)
var_p = [c for c in varying_hparams(ppo_b) if c.startswith("hp_")]
marg_rows = []
for c in var_p:
    g = last_p.groupby(c, dropna=False)["mean_reward"].agg(["mean", "std", "count"]).reset_index()
    g.insert(0, "hp_col", c)
    g = g.rename(columns={c: "hp_value"})
    marg_rows.append(g)
if marg_rows:
    marginal = pd.concat(marg_rows, ignore_index=True).sort_values(["hp_col", "mean"], ascending=[True, False])
    display(marginal)

fig, ax = plt.subplots(figsize=(11, 5))
plot_runs_curves(ppo_b, ax=ax, title="PPO/Bobail — mean_reward par run_dir")
plt.tight_layout()

n_p = len(last_p)
n_hi = (last_p["mean_reward"] >= 0.94).sum()
print(f"PPO Bobail dernier ckpt : {n_hi}/{n_p} runs avec mean_reward >= 0.94")"""),
    cell("markdown", """## 6. Famille planning

`mcts` et `random_rollout` planifient au coup suivant avec un budget de simulations ; le principal réglage est le budget (trade-off qualité / latence)."""),
    cell("markdown", """### 6.1 Effet du budget sur les performances"""),
    cell("code", """plan = dfl.query("agent in ['mcts', 'random_rollout']")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for ax, env in zip(axes.flat, ENV_LIST):
    sub = plan.query("env == @env")
    if sub.empty:
        ax.set_visible(False)
        continue
    plot_hp_curves(sub, "agent", "mean_reward", ax=ax,
                   title=f"{env} — mean_reward vs budget")
    ax.set_xlabel("budget (n_simulations)")
plt.tight_layout()"""),
    cell("markdown", """Sur **line_world** et **grid_world**, les deux agents atteignent vite le score maximal — peu discriminant. Sur **TicTacToe**, MCTS et Random Rollout restent proches sur la plage de budgets. Sur **Bobail**, Random Rollout peut atteindre très tôt un score maximal alors que MCTS progresse avec le budget."""),
    cell("markdown", """### 6.2 MCTS vs Random Rollout — performance et coût au budget maximal testé"""),
    cell("code", """plan_last = plan.sort_values("eval_axis_value").groupby("run_dir").tail(1)
perf = (plan_last.pivot_table(index="env", columns="agent",
                               values="mean_reward", aggfunc="mean").round(3))
cost = (plan_last.pivot_table(index="env", columns="agent",
                               values="mean_action_time_ms", aggfunc="mean").round(1))

comb = perf.add_suffix("_reward").join(cost.add_suffix("_ms"), how="outer")
print("mean_reward et temps moyen par coup (ms) au dernier budget par run :")
display(comb)

fig, ax = plt.subplots(figsize=(8, 3))
perf.plot.barh(ax=ax, color=[FAMILY_COLOR["planning"], "#A5D6A7"], alpha=0.85, edgecolor="black")
ax.set_xlabel("mean_reward (budget max)")
ax.set_title("MCTS vs Random Rollout — performance au budget maximal")
plt.tight_layout()"""),
    cell("markdown", """Random Rollout peut dépasser MCTS sur Bobail au prix d'une latence par coup nettement plus élevée ; sur des grilles simples, MCTS peut être préférable si la latence est contrainte."""),
    cell("markdown", """## 7. Synthèse multi-environnements & implications (soutenance)

Table : au **dernier** point d'évaluation par `run_dir`, meilleur agent par famille (value-based, policy-gradient, planning) et score tabulaire (moyenne des runs `tabular_q` si plusieurs)."""),
    cell("code", """VALUE_AGENTS = ["dqn", "ddqn", "ddqn_er", "ddqn_per"]
PG_AGENTS = ["reinforce", "reinforce_mean_baseline", "reinforce_critic", "ppo"]
PLAN_AGENTS = ["mcts", "random_rollout"]

last_all = dfl.sort_values("eval_axis_value").groupby("run_dir").tail(1)


def best_in_family(sub, agents, metric="mean_reward"):
    s = sub[sub["agent"].isin(agents)]
    if s.empty:
        return None, np.nan
    g = s.groupby("agent")[metric].mean().sort_values(ascending=False)
    return g.index[0], float(g.iloc[0])


rows = []
for env in ENV_LIST:
    le = last_all[last_all["env"] == env]
    vb_a, vb_r = best_in_family(le, VALUE_AGENTS)
    pg_a, pg_r = best_in_family(le, PG_AGENTS)
    pl_a, pl_r = best_in_family(le, PLAN_AGENTS)
    tq = le[le["agent"] == "tabular_q"]["mean_reward"]
    tq_r = float(tq.mean()) if len(tq) else np.nan
    rows.append({
        "env": env,
        "best_value_based": f"{vb_a} ({vb_r:.3f})" if vb_a else "",
        "best_policy_gradient": f"{pg_a} ({pg_r:.3f})" if pg_a else "",
        "best_planning": f"{pl_a} ({pl_r:.3f})" if pl_a else "",
        "tabular_q_mean": f"{tq_r:.3f}" if np.isfinite(tq_r) else "",
    })
syn = pd.DataFrame(rows)
display(syn)"""),
    cell("markdown", """**Implications (quelques phrases pour slides)**

1. **HP non indépendants** — balayer « un paramètre à la fois » sur des YAML qui changent plusieurs champs mène à des conclusions biaisées ; préférer courbes par `run_dir` ou grilles complètes.
2. **Tabulaire vs deep value-based** — sur Bobail, le Q-tabulaire exploite mieux l'espace fini d'états que les DQN testés ici (épisodes courts vs troncature).
3. **Policy gradient** — très forts sur Bobail dans ce protocole ; ne pas généraliser sans vérifier chaque environnement (ex. TicTacToe plus dur pour PPO).
4. **Planning** — budget = curseur score/latence ; RR coûte souvent plus cher par coup que MCTS pour un gain parfois marginal.
5. **Variance** — peu de seeds et recherche manuelle : les scores sont des **bornes inférieures** réalistes, pas des optimums globaux."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "python3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11.0"}}, "nbformat": 4, "nbformat_minor": 5}
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Wrote", NB, "cells:", len(cells))
