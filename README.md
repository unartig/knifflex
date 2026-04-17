# knifflex

Have you also wanted to become better in [Kniffel](https://de.wikipedia.org/wiki/Kniffel) (engl. [Yahtzee](https://en.wikipedia.org/wiki/Yahtzee)), or even the bestest?
Are the probabilities involved too confusing, and unclear which category to prioritize given a game state or maybe what the optimal strategy is?
Kniffel can be solved analytically, but turns out we don't gain anything from that, the strategy is still a black-box!

Say no more, **Knifflex** is here to help!

All it takes is a perfect understanding of transition probabilities and some slight reweighting of expected returns ;^)  

**Knifflex** is basically a bot, that is trained gradient free, using either **[evolutionary strategies (ES)](https://arxiv.org/pdf/1703.03864)** or a **[genetic island model*](https://link.springer.com/chapter/10.1007/BFb0027170)*.
Built with [JAX](https://github.com/google/jax) and [Equinox](https://github.com/patrick-kidger/equinox), fully JIT-compiled.

And the nicest part is, **Kniffel** is directly interpretable and will teach you how to become better at Kniffel.


This project took me ~4 years, with numerous highs and lows and wildly different methodologies I have tried and failed with, until I came up with the current implementation!
Shout out to the [Yahtzotron](https://github.com/dionhaefner/yahtzotron) a very similar project.

---

## How it works

The agent plays Kniffel by scoring each possible action using a learned weight matrix applied to a compact game-state context.
No neural network, no value function approximation!
Simply a linear scoring rule evolved to be surprisingly good, reaching ~244 avg. score compared to 245.8 optimal.

### The Genome

Each genome learns two weight matrices `W` and `W_scale` of shape `(13 categories × 17 context dims)` that map the current game state to a score for each category.
There are two variants:

**`FullWGenome`** — stores `W` directly as a `(13, 17)` matrix.

**`DecompWGenome`** (default) — factorizes each matrix `W = A @ B` with a low rank (rank 1 by default), giving a much smaller parameter count.
This acts like a bottleneck: each category's preference is a scalar multiple of a single shared "feature direction".

A scalar `bonus_uplift` is additionally learned: it adds a bonus to upper-section categories when scoring them would bring the agent closer to the 63-point bonus threshold.

**At inference**, the genome's `oracle_action` function:
1. Builds a 17-dim context vector from the current state (scores, upper sum, bonus distance,rolls left and round, everything normalized to [0, 1]).
2. Computes per-category weights and scales via `adjusted_ev = (raw_ev + bonus_uplift) * softplus(W_scale @ ctx) + (W @ ctx)`.
  - Uses a precomputed **EV table** (expected value of each dice configuration for each scoring category at each roll depth) to evaluate all 252 possible dice configurations.
3. For rerolls: multiplies the **transition table** (probability of reaching each dice config from the current one given a keep mask) against future utilities to find the best keep mask.
4. Returns either a scoring action or the best reroll mask.

---

## Training

Two training approaches are provided:

### `es.py` — Evolution Strategies

A single genome is trained using ES (OpenAI ES style), a gradient free technique to estimate pseude-gradients.
Each step the genome is copied and perturbed using **antithetic noise** (half normal, half negated — reducing variance).
Now we can estimate the pseudo-gradients via `mean(fitness_norm * noise) / sigma`.
All that is missing is now to apply updates to the genome, here we use Adam.
This is gradient-free in the game sense but uses a smooth gradient estimator over the noise distribution.

### `train_w.py` — Island Genetic Algorithm

A more elaborate evolutionary setup with a number of islands, with some number genomes each, plus a wildcard island!
Each step and for each island a number of elites/survivors is chosen via _k-way tournament selection_, creating survival pressure.
Following, the island populations are mutated (small random pertubations) and crossed-over (creates an offspring combining features of two parents) keeping population diversity up.

Every X'th step, the best individuals of an island migrate in a ring-style to the next island, replacing their worst.
A _wildcard_ island is completely reseeded every migration turn with highly random genomes, injecting diversity in the global population.

Right now the genetic algorithm learns slower and does reach a lower average score compared to the evolutionary strategy.
Tuning hyperparameters might help!

---

## Project structure

```
./
├── data/                   # directory for precomputed things / trained genomes
│   └── runs/               # training progress and genome-checkpoints
│
└── knifflex/               # code
    ├── genome/             # everything regarding the genome (definition + serialization + training)
    ├── game/               # game logic - computation of transition tensor / expected value matrix
    ├── utils/              # well...
    └── ui/                 # TUI / frontend code
 
```

## Getting started

```bash
pip install -r requirements.txt
pip install -e .
```

## Running

```bash
# Train with Evolution Strategies
python -m knifflex.genome.evolutionary_strategy

# Train with Island Genetic Algorithm
python -m knifflex.genome.island_genetic_algorithm

# Watch training in TensorBoard
tensorboard --logdir data/runs/

# Play Kniffel and get help from a already trained Bot
python -m knifflex.ui.frontend
```

## Frontend

Currently looks like the following:
```
  🎲 KNIFFEL | Round: 1/13 | Rolls: 2

  ~~~~~~~~~~~ ~~~~~~~~~~~ ┌─────────┐ ┌─────────┐ ┌─────────┐
  ~         ~ ~ X     X ~ │ ●     ● │ │ ●     ● │ │ ●     ● │
  ~    X    ~ ~         ~ │         │ │ ●     ● │ │ ●     ● │
  ~         ~ ~ X     X ~ │ ●     ● │ │ ●     ● │ │ ●     ● │
  ~~~~~~~~~~~ ~~~~~~~~~~~ └─────────┘ └─────────┘ └─────────┘

  SCORECARD                    RAW EV │ ADJ. EV     DECISION LOGIC
    Ones              ---         1.1 │  5094.6
    Twos              ---         2.2 │  4032.5     Current Val: 5681.29
    Threes            ---         3.3 │  3066.3     Reroll  Val: 7677.78
    Fours             ---         7.8 │  4490.0
    Fives             ---         5.6 │   486.8     Given the current reroll selection:
    Sixes             ---        16.7 │  7677.8     » SHOULD REROLL
    Full House        ---         6.7 │  4174.7
    3 of a Kind       ---        15.9 │  5395.3     💡 AI SUGGESTS: Reroll [1, 1, 1, 0, 0]
    4 of a Kind       ---         4.6 │  5079.1     1 reroll - 0 keep
    Small Straight    ---        10.1 │  2966.7
    Large Straight    ---         2.8 │  2324.1     Mode: Bonus Hunt
    Chance            ---        24.5 │  6948.8
    Kniffel           ---         0.8 │  4760.0     Bias: BonusDist

  WASD/HJKL: Move | SPACE: Toggle | ENTER: Confirm
  TAB: Switch Mode | G: Toggle AI | Q: Quit
```
I hope I will find the time to improve the Genome interpretability and Suggestions in the near futu in the near future!

