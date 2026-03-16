# docs/IMPLEMENTATION_GUIDE.md
# Syllabus Projet — 2025-2026

**Année :** 2025-2026  
**Imprimé le :** 10/03/26 19:53

---

## Matières, formations et groupes

| Champ | Valeur |
|---|---|
| **Matière liée au projet** | M2 - t2 - deep reinforcement learning |
| **Formation** | 2026-5A-IABD-DRL |
| **Enseignant(s)** | VIDAL Nicolas |
| **Email(s)** | nvidal@myges.fr |
| **Nombre d'étudiant par groupe** | 3 |
| **Règles de constitution des groupes** | Imposé |
| **Type de sujet** | Imposé |
| **Charge de travail estimée par étudiant** | 30,00 h |

---

## Sujet(s) du projet

### Objectif du projet

*(À la fin du projet les étudiants sauront réaliser un…)*

- Évaluer les différentes techniques d'apprentissage par renforcement profond vues en cours sur **1 nouvel environnement** parmi la liste ci-dessous (en plus des précédents).
- Comprendre les différents atouts de chaque algorithme et savoir quand les appliquer.

---

## Descriptif détaillé

> ⚠️ **ATTENTION : TOUTE IMPOSSIBILITÉ POUR LE GROUPE D'ÉTUDIANT D'EXPLIQUER LE CODE UTILISÉ POUR LE PROJET CONDUIRA À L'INVALIDATION DU PROJET.**

### Environnements de départ

- **Pour tests :** Line World
- **Pour tests :** Grid World
- **Pour tests :** TicTacToe versus Random

**\+ 1 au choix parmi :**

- **Quarto** (vs Random ou Heuristique)
  - https://boardgamearena.com/gamepanel?game=quarto
- **Bobail** (vs Random ou Heuristique)
  - https://boardgamearena.com/gamepanel?game=bobail
- **Pond** (versus Random ou Heuristique)
  - https://boardgamearena.com/gamepanel?game=pond
- **ColorPop** (Solo ou Vs Randoms ou Heuristiques)
  - https://boardgamearena.com/gamepanel?game=colorpop

---

### Types d'agents à étudier

1. Random
2. TabularQLearning *(quand possible)*
3. DeepQLearning
4. DoubleDeepQLearning
5. DoubleDeepQLearningWithExperienceReplay
6. DoubleDeepQLearningWithPrioritizedExperienceReplay
7. REINFORCE
8. REINFORCE with mean baseline
9. REINFORCE with Baseline Learned by a Critic
10. PPO A2C style
11. RandomRollout
12. Monte Carlo Tree Search (UCT)
13. Expert Apprentice
14. Alpha Zero
15. MuZero
16. MuZero stochastique

---

### Métriques à obtenir

> ⚠️ Attention : métriques pour la **policy obtenue**, pas pour la policy en mode entraînement.

**Score moyen (pour chaque agent) au bout de :**

- 1 000 parties d'entraînement
- 10 000 parties d'entraînement
- 100 000 parties d'entraînement
- 1 000 000 parties d'entraînement *(si possible)*
- XXX parties d'entraînement *(si possible)*

**Temps moyen mis pour exécuter un coup**

**Si la partie est de durée variable — Longueur moyenne (nombre de steps) d'une partie au bout de :**

- 1 000 parties d'entraînement
- 10 000 parties d'entraînement
- 100 000 parties d'entraînement
- 1 000 000 parties d'entraînement *(si possible)*
- XXX parties *(si possible)*

---

### Interface graphique

Il sera également nécessaire de présenter une **interface graphique** permettant de :
- Regarder jouer chaque agent
- Mettre à disposition un agent **"humain"**

---

### Code et modèles

Pour chaque environnement et chaque algorithme, les étudiants devront étudier les performances de l'algorithme et retranscrire leurs résultats.

Les étudiants devront fournir :
- L'**intégralité du code** leur ayant permis d'obtenir leurs résultats
- Les **modèles entraînés et sauvegardés** (keras / tensorflow / pytorch / jax / keras_core / burn) prêts à être exécutés pour confirmer les résultats présentés

---

### Rapport et présentation

Les étudiants devront présenter ces résultats dans un **rapport** où devront figurer :
- L'évolution des métriques d'apprentissage au cours de l'entraînement
- Une **présentation**

Dans ces derniers, les étudiants devront :
- Faire valoir leur **méthodologie de choix d'hyperparamètres**
- Proposer leur **interprétation des résultats obtenus**

---

## Outils informatiques à installer

- TensorFlow / Keras
- PyTorch
- JAX / Keras Core
- Burn

---

## Ouvrages de référence

- *Reinforcement Learning: An Introduction* — Richard S. Sutton and Andrew G. Barto

---

## Présentation

| Champ | Valeur |
|---|---|
| **Durée de présentation par groupe** | 20 min |
| **Audience** | À huis clos |
| **Type de présentation** | Présentation / PowerPoint |

---

## Livrables et étapes de suivi

### Étape intermédiaire — *Dimanche 29/03/2026, 23h59*

**1er Gameplay choisi implémenté** (ainsi que les environnements de tests) :

- Simulation de jeu avec joueur random *(calculer le nombre de parties / seconde)*
- Jeu avec joueur humain + GUI
- Proposition de description de l'état du jeu *(vecteur d'encoding)*
- Proposition de description d'une action du jeu *(vecteur d'encoding)*

**Rendu :** *(tout sur MyGES, pas simplement un lien git)*

- Code source
- Démonstration rapide
- Documents de spécifications *(encoding vectors)*

---

### Rendu final — *Dimanche 26/04/2026, 23h59*

**Rendu :** *(tout sur MyGES, pas simplement un lien git)*

- Code source
- Démonstration rapide
- Readme de reproduction des résultats / lancement démo
- Rapport conséquent sur les expérimentations menées / résultats obtenus et observations critiques de ces résultats
- Documents de résultats *(metrics sur les gameplays avec tous les algos)*
- Slides de présentation utilisés pour la soutenance

---

### Soutenance

- Présentation orale de 20 minutes
- À huis clos
