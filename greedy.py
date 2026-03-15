from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import trange

from game import KniffelEnv


class GreedyKniffler:
    def __init__(self, *, verbose: bool =True, initialize: bool = True) -> None:
        self.verbose = verbose

        reroll_maps = list(product([0, 1], repeat=5))
        self.reroll_lists = np.array(reroll_maps, dtype=bool)
        self.reroll_ints = np.array([int("".join(map(str, rm))) for rm in reroll_maps])

        # der bot darf die env nicht speichern, nur zum aufbau der matrizen nutzen
        env = KniffelEnv()
        self.case_names = env.case_names
        self.case_names_dict = {cn: i for i, cn in enumerate(self.case_names)}
        self.df, self.obere_treffer, self.obere_base = self.init_df(env)  # wurfints idx cases cols
        self.np_df = self.df.to_numpy()
        self.alive_cats = self.df.columns.to_list()
        self.alive_cat_ind = np.arange(0, 13, 1)
        self.dfcp = self.df.copy()
        # list of ints
        self.roll_ints = self.df.index.to_numpy()
        if not (Path("roll_lookups.npy").exists() and Path("possible_rolls.npy").exists()) or initialize:
            self._roll_lookup, self._possible_rolls = self.init_roll_lookup()
            np.save("roll_lookups.npy", self._roll_lookup, allow_pickle=True)
            np.save("possible_rolls.npy", self._possible_rolls, allow_pickle=True)
        else:
            self._roll_lookup, self._possible_rolls = np.load("roll_lookups.npy"), np.load("possible_rolls.npy")
        self.roll_lookup, self.possible_rolls = self._roll_lookup.copy(), self._possible_rolls.copy()

        self.tmp = np.divide(
            (self.obere_treffer - 1 * self.obere_base), 3 * self.obere_base, where=self.obere_base != 0
        )  # TODO treffer damping

    def reset(self):
        self.alive_cat_ind = np.arange(0, 13, 1)
        self.alive_cats = self.df.columns.to_list()
        return self

    def wurf_to_int(self, roll: list[int] | np.ndarray) -> int:
        return int("".join(map(str, list(roll))))

    def init_df(self, env):
        temp = [list(range(1, 7)) for _ in range(5)]
        permut = list(product(*temp))

        wurf_ints = []
        for i in permut[:]:
            si = list(np.sort(i))
            si = int("".join(map(str, si)))
            if si not in wurf_ints and si > 10000:
                wurf_ints.append(si)
        wurf_lists = np.array(wurf_ints)

        df = pd.DataFrame(index=wurf_lists[:], columns=self.case_names, dtype=np.float32)
        obere_treffer = np.zeros((32, 252, 13))
        obere_base = np.zeros((32, 252, 13))
        for i, kat in enumerate(df.columns):
            for wurf_index, wurf_int in enumerate(wurf_ints):
                wurf_liste = np.array([int(d) for d in str(wurf_int)])
                df.loc[wurf_int, kat] = env.cases[i](wurf_liste)
                if i < 6:
                    obere_base[:, wurf_index, i] = df.loc[wurf_int, kat] / (i + 1)
                    obere_treffer[:, wurf_index, i] = df.loc[wurf_int, kat]
        # df["augenzahl"] *= 0.2
        # df["kniffel"] *= 1.1
        if self.verbose:
            print(df.head(20))
        return df, obere_treffer, obere_base

    def init_roll_lookup(self) -> tuple[np.ndarray, np.ndarray]:
        # TODO: Überprüfen
        expects = {i: (i / 6**i) * np.clip(self.np_df, 0, 1) for i in range(6)}
        expects[0] = np.clip(self.np_df, 0, 1)

        # wurf x mask x wurf x kategorie
        empty_expects = np.zeros((252, 32, 252, 13), dtype=np.float32)
        possible_rolls = np.zeros((252, 32, 252), dtype=np.bool)
        for i in trange(self.roll_ints.shape[0]):
            wurf_i = np.array([int(d) for d in str(self.roll_ints[i].copy())], dtype=np.int8)  # to array
            for j in range(self.roll_ints.shape[0]):
                wurf_j = np.array([int(d) for d in str(self.roll_ints[j].copy())], dtype=np.int8)

                possibe_rolls = self.get_rmasks(wurf_i, wurf_j)
                for roll in possibe_rolls:
                    roll_int = self.wurf_to_int(roll * 1)
                    roll_index = np.where(self.reroll_ints == roll_int)
                    empty_expects[i, roll_index, j] += np.array(expects[np.sum(roll)][j], dtype=np.float32)
                    possible_rolls[i, roll_index, j] = 1.0
        expects = empty_expects

        if self.verbose:
            df_cp = pd.DataFrame(expects[0, 0], index=self.roll_ints, columns=self.df.columns)
            print(f"Distance for 1 1 1 1 1 reroll with reroll pattern 0 0 0 0 0\n{df_cp}")
            df_cp = pd.DataFrame(expects[0, 1], index=self.roll_ints, columns=self.df.columns)
            print(f"Distance for 1 1 1 1 1 reroll with reroll pattern 0 0 0 0 1\n{df_cp}")
        return expects, possible_rolls

    def get_rmasks(self, roll: np.ndarray, target_roll: np.ndarray) -> list[np.ndarray]:
        target_counts = np.bincount(target_roll, minlength=7)

        # kept dice for each mask
        kept = roll[None, :] * (~self.reroll_lists)

        # count per mask
        kept_counts = np.apply_along_axis(
            lambda x: np.bincount(x[x > 0], minlength=7),
            axis=1,
            arr=kept,
        )

        valid = np.all(kept_counts <= target_counts, axis=1)
        return self.reroll_lists[valid]

    def get_random_action(self, act_env):
        valid_actions = np.arange(0, act_env.n)
        action = np.random.choice(valid_actions)
        return action

    def get_value(self, env, act_env, reroll=True):
        for regel in self.regeln:
            action, wert = regel.check_action(env)
            vwert = 0
            if wert > vwert and wert in act_env:
                return action, wert, regel
        return -1, -1, None

    def get_action(self, wurf, runde=None, subrunde=None, spielplan=None):
        # top are goal
        wurf_zahl = self.wurf_to_int(wurf)
        wurf_index = np.where(self.roll_ints == wurf_zahl)[0][0]

        roll_mask = np.zeros((1, 5))
        if subrunde == 0:
            roll_mask = self.one_roll(wurf_zahl, wurf_index)
        if subrunde == 1:
            roll_mask = self.one_roll(wurf_zahl, wurf_index)
        if np.sum(roll_mask) != 0:
            return True, roll_mask
        # kat kreuzen
        expectency = self.df.loc[wurf_zahl, :][self.alive_cats]
        katego_name = expectency.T.idxmax()
        # kreuze weil wurf kacke
        if (
            self.df.loc[wurf_zahl, katego_name]
            < 0.15 * self.df[katego_name].where(self.df[katego_name] != 0).mean()
        ):  # TODO schlechte kreuzen gain
            katego_name = self.df[self.alive_cats].mean().idxmin()

        case_index = self.case_names_dict[katego_name]

        self.alive_cats.remove(katego_name)
        self.alive_cat_ind = self.alive_cat_ind[self.alive_cat_ind != case_index]
        if self.verbose:
            print(f"Bot möchte {katego_name} für {wurf} kreuzen, erwarteter gewinn {expectency[katego_name]}")
        return False, case_index

    def one_roll(self, wurf_zahl, wurf_index, return_exp=False):
        # rool_lookup shape = 252, 32, 252, 13
        tmp_expect = self.roll_lookup[wurf_index][:, :, self.alive_cat_ind]
        tmp_expect = np.multiply(tmp_expect, self.np_df[:, self.alive_cat_ind])

        # find max reroll_pattern
        # max cat for each dice combination
        maxs = np.max(tmp_expect, axis=2)
        # mean return for each reroll pattern
        tmp_means = np.mean(maxs, axis=-1)

        max_mean_indx = np.argmax(tmp_means, axis=0)

        exp = tmp_expect[max_mean_indx, :]

        roll_mask = self.reroll_lists[max_mean_indx]

        if self.verbose:
            expectency = tmp_expect[max_mean_indx]
            expectency = pd.DataFrame(expectency, self.df.index, self.df.columns[self.alive_cat_ind], dtype=np.float32)
            expectency = expectency[self.alive_cats]
            sorted_means = expectency.mean().sort_values(ascending=False).index[:3].to_list()
            gewollter_wurf = np.argmax(np.max(exp.T, axis=1))
            gewollter_wurf = np.array([int(d) for d in str(gewollter_wurf)])
            print(
                f"Bot möchte rerollen: von {wurf_zahl} zu "
                f"{gewollter_wurf} mit Maske {roll_mask},"
                f"mit Auge auf {sorted_means}"
            )
        if return_exp:
            # print(np.max(exp), exp.shape, roll_mask)
            return np.max(exp)
        return roll_mask

    def two_rolls(self, wurf_zahl, wurf_index):
        tmp_poss = self.possible_rolls[wurf_index]

        max_exp = 0
        max_roll_indx = 0
        for roll_indx, poss_wurfs in enumerate(tmp_poss):
            roll_mask = self.reroll_lists[roll_indx]
            roll_prob = np.sum(roll_mask) / 6 ** np.sum(roll_mask)
            roll_prob = 1 if roll_indx == 0 else roll_prob
            mean_exp = 0
            poss_count = 0
            for rwurf_indx, poss in enumerate(poss_wurfs):
                if poss:
                    tmp_exp = self.one_roll(self.roll_ints[rwurf_indx], rwurf_indx, return_exp=True)
                    mean_exp += tmp_exp * roll_prob
                    poss_count += 1
            # print(wurf_zahl, self.reroll_lists[roll_indx], mean_exp)
            # mean_exp /= poss_count
            if mean_exp > max_exp:
                max_exp = mean_exp
                max_roll_indx = roll_indx

        return self.reroll_lists[max_roll_indx]

    def force_action(self, case_index):
        if case_index in self.alive_cat_ind:
            self.alive_cat_ind = self.alive_cat_ind[self.alive_cat_ind != case_index]

    def get_roll_mask(self, wurf, gewollter_wurf):
        roller = []
        for w in wurf:
            if w in gewollter_wurf:
                roller.append(False)
                index_to_remove = np.argmax(gewollter_wurf == w)
                gewollter_wurf = np.delete(gewollter_wurf, index_to_remove)
            else:
                roller.append(True)

        return np.array(roller)


if __name__ == "__main__":
    import time

    # Single Core Games
    verbose = True
    # tenv = game.KniffelEnv()
    bot = GreedyKniffler(verbose=verbose, initialize=False)

    env = KniffelEnv(retry_on_wrong_action=True, verbose=verbose)
    # env.seed = seed
    start = time.time() if verbose else None
    start = time.time()
    ts = start
    running = 0
    _sum = 0

    n_rounds = 1
    boni = 0
    while running < n_rounds:
        wurf, runde, subrunde, spielplan, ende = env.get_observation()

        if ende:
            end = time.time()
            if verbose:
                print("Single round in", end - ts, "s")
            ts = end
            _sum += env.punkte
            boni += 1 if env.bonus > 0 else 0

            running += 1
            if running == n_rounds:
                break
            env = env.reset(incr_rand_seed=True)

            bot = bot.reset()
            wurf, runde, subrunde, spielplan, _ = env.get_observation()

        reroll, action = bot.get_action(wurf, runde, subrunde, spielplan)
        if not reroll:
            env.kreuze_kategorie(action, by_name=False)  # action name
        else:
            env.reroll(action)
    mean = _sum / n_rounds
    print(f"{n_rounds} rounds in", end - start, "s")
    print(f"Mean Return {mean}, {boni}*bonus")
