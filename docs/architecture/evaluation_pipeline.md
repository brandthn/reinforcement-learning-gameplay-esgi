# Pipeline d'Evaluation

## Objectif

L'evaluateur mesure la performance d'un agent **en mode inference** (pas d'exploration, politique gelee) sur N parties.

Le sujet demande specifiquement :
- Score moyen a 1k, 10k, 100k episodes d'entrainement
- Temps moyen pour executer un coup
- Longueur moyenne d'une partie

---

## Metriques collectees

| Metrique | Description | Colonne CSV |
|----------|-------------|-------------|
| **Mean reward** | Recompense moyenne sur N parties | `mean_reward` |
| **Std reward** | Ecart-type de la recompense | `std_reward` |
| **Mean steps** | Longueur moyenne d'une partie | `mean_steps` |
| **Std steps** | Ecart-type de la longueur | `std_steps` |
| **Mean action time (ms)** | Temps moyen par action en millisecondes | `mean_action_time_ms` |
| **Std action time (ms)** | Ecart-type du temps par action | `std_action_time_ms` |

---

## Sequence d'evaluation : Single-Player

```mermaid
sequenceDiagram
    participant EV as Evaluator
    participant E as Environment
    participant A as Agent

    loop N = num_games parties
        EV->>E: reset()
        E-->>EV: state

        loop max 10000 steps
            EV->>E: available_actions()
            EV->>EV: t0 = time.perf_counter()
            EV->>A: act(state, available, training=False)
            Note over A: PAS d'exploration<br/>Politique greedy pure
            A-->>EV: action
            EV->>EV: elapsed = time.perf_counter() - t0
            EV->>E: step(action)
            E-->>EV: (state, reward, done)
        end

        EV->>EV: Stocker (reward, steps, action_times)
    end

    EV->>EV: Calculer moyennes et ecarts-types
```

## Sequence d'evaluation : Adversarial

```mermaid
sequenceDiagram
    participant EV as Evaluator
    participant E as Environment
    participant A as Agent (J0)
    participant O as Opponent (J1)

    loop N parties
        EV->>E: reset()

        loop Jusqu'a done
            alt Joueur 0 (agent evalue)
                EV->>EV: Mesurer le temps
                EV->>A: act(state, available, training=False)
                A-->>EV: action
                EV->>E: step(action)
                E-->>EV: (state, reward, done)
                EV->>EV: agent_reward += reward
            else Joueur 1 (opponent)
                EV->>O: act(state, available, training=False)
                O-->>EV: action
                EV->>E: step(action)
                E-->>EV: (state, reward, done)
                alt done et opponent a gagne
                    EV->>EV: agent_reward -= reward
                end
            end
        end
    end
```

### Point important : seul le temps de l'agent evalue est mesure

Le temps de l'adversaire n'est **pas** inclus dans les metriques de performance.

---

## Quand l'evaluation est-elle executee ?

```mermaid
graph LR
    A["Episode 1"] --> B["..."]
    B --> C["Episode 1000"]
    C -->|"Checkpoint!"| D["Evaluate 100 games"]
    D --> E["Sauver model_1000.pt"]
    E --> F["..."]
    F --> G["Episode 10000"]
    G -->|"Checkpoint!"| H["Evaluate 100 games"]
    H --> I["Sauver model_10000.pt"]
    I --> J["..."]
    J --> K["Episode 100000"]
    K -->|"Checkpoint!"| L["Evaluate 100 games"]
    L --> M["Sauver model_100000.pt"]

    style D fill:#FF9800,color:#fff
    style H fill:#FF9800,color:#fff
    style L fill:#FF9800,color:#fff
```

Les checkpoints sont configurables dans le YAML :
```yaml
eval:
  checkpoints: [1000, 10000, 100000]
  num_games: 100
```

---

## Promotion du meilleur modele

Le script `scripts/promote_best.py` selectionne le **meilleur checkpoint** par combinaison (env, agent) :

```mermaid
graph TD
    A["Scan results/"] --> B["Pour chaque env/agent"]
    B --> C["Lire tous les metrics.csv"]
    C --> D["Trier par mean_reward"]
    D --> E["Copier le meilleur vers<br/>results/{env}/{agent}/best/"]

    style E fill:#4CAF50,color:#fff
```

```bash
uv run python scripts/promote_best.py
```

Resultat :
```
results/{env}/{agent}/best/
├── model.pt       # Meilleur modele
└── config.yaml    # Configuration associee
```

La GUI charge en priorite les modeles depuis `best/`.

---

## Re-evaluation post-entrainement

```bash
uv run python scripts/evaluate_all.py
```

Ce script re-evalue tous les modeles sauvegardes et ecrit un fichier `metrics_reeval.csv` dans chaque dossier de run. Utile pour comparer les performances avec des parametres d'evaluation differents.
