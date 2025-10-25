"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from loguru import logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass


# ===== Experiment settings =====
Bandit_Reward = [1, 2, 3, 4]   # true means of each arm
NumberOfTrials = 20000
np.random.seed(42)             # reproducible runs

# Thompson Sampling (Normal rewards with known variance)
TS_KNOWN_PRECISION = 1.0       # tau = 1/sigma^2  -> sigma=1
TS_PRIOR_MEAN = 0.0
TS_PRIOR_VAR = 10.0



class Bandit(ABC):
    """ """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        """ """
        pass

    @abstractmethod
    def update(self):
        """ """
        pass

    @abstractmethod
    def experiment(self):
        """ """
        pass

    @abstractmethod
    def report(self):
        """ """
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#
@dataclass
class ArmState:
    """ """
    true_mean: float
    n: int = 0
    mean_est: float = 0.0
    sum_x: float = 0.0

    def observe(self, x: float):
        """

        Args:
          x: float: 

        Returns:

        """
        self.n += 1
        self.sum_x += x
        self.mean_est += (x - self.mean_est) / self.n


class Visualization():
    """ """

    def plot1(self, df_algo, title_prefix="Algorithm"):
        """Per-arm learning curves for a single algorithm.
        Shows cumulative average reward of each arm over time (linear & log x).

        Args:
          df_algo: 
          title_prefix:  (Default value = "Algorithm")

        Returns:

        """
        df = df_algo.copy()
        df["cum_reward"] = df.groupby("Bandit")["Reward"].cumsum()
        df["cum_count"]  = df.groupby("Bandit").cumcount() + 1
        df["cum_avg"]    = df["cum_reward"] / df["cum_count"]

        # Linear x-axis
        plt.figure(figsize=(10, 5))
        for arm in sorted(df["Bandit"].unique()):
            sub = df[df["Bandit"] == arm]
            true_mu = sub["TrueMean"].iloc[0]
            plt.plot(sub["t"], sub["cum_avg"], label=f"Arm {arm} (μ*={true_mu:.1f})")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative avg reward")
        plt.title(f"{title_prefix}: Per-arm learning (linear)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Log x-axis
        plt.figure(figsize=(10, 5))
        for arm in sorted(df["Bandit"].unique()):
            sub = df[df["Bandit"] == arm]
            true_mu = sub["TrueMean"].iloc[0]
            plt.plot(sub["t"], sub["cum_avg"], label=f"Arm {arm} (μ*={true_mu:.1f})")
        plt.xscale("log")
        plt.xlabel("Trial (log)")
        plt.ylabel("Cumulative avg reward")
        plt.title(f"{title_prefix}: Per-arm learning (log x)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot2(self, df_eg, df_ts):
        """Compare cumulative rewards and cumulative regrets across algorithms.

        Args:
          df_eg: 
          df_ts: 

        Returns:

        """
        # Rewards
        plt.figure(figsize=(10, 5))
        plt.plot(df_eg["t"], df_eg["Reward"].cumsum(), label="Epsilon-Greedy")
        plt.plot(df_ts["t"], df_ts["Reward"].cumsum(), label="Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative reward")
        plt.title("Cumulative Rewards: EG vs TS")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Regrets
        plt.figure(figsize=(10, 5))
        plt.plot(df_eg["t"], df_eg["Regret"].cumsum(), label="Epsilon-Greedy")
        plt.plot(df_ts["t"], df_ts["Regret"].cumsum(), label="Thompson Sampling")
        plt.xlabel("Trial")
        plt.ylabel("Cumulative regret")
        plt.title("Cumulative Regret: EG vs TS")
        plt.legend()
        plt.tight_layout()
        plt.show()


#--------------------------------------#

class EpsilonGreedy(Bandit):
    def __init__(self, p, epsilon0: float = 0.5):
        """Initialize an epsilon-greedy solver with decaying epsilon (ε_t = ε0 / t).

        Args:
            p (list[float]): True means for each arm (e.g., [1, 2, 3, 4]).
            epsilon0 (float, optional): Initial exploration rate. Actual epsilon at
                time t is epsilon0 / t. Defaults to 0.5.
        """
        self.true_means = list(p)
        self.k = len(self.true_means)
        self.epsilon0 = float(epsilon0)
        self.t = 0  # trial counter

        # one ArmState per arm to track estimates
        self.arms = [ArmState(m) for m in self.true_means]

        # logging containers
        self.rows = []
        self.df = None

        # for regret calculation
        self.best_mean = max(self.true_means)

    def __repr__(self):
        return f"EpsilonGreedy(k={self.k}, epsilon0={self.epsilon0}, t={self.t})"

    def _sample_reward(self, mean: float, precision: float = 1.0) -> float:
        """Draw a Normal reward with known precision τ (default τ=1 → σ=1).

        Args:
          mean: float: 
          precision: float:  (Default value = 1.0)

        Returns:

        """
        sigma = (1.0 / precision) ** 0.5
        return np.random.normal(loc=mean, scale=sigma)

    def pull(self):
        """Choose an arm using ε-greedy with decay ε_t = ε0 / t.
        Returns: (arm_index, observed_reward)

        Args:

        Returns:

        """
        self.t += 1
        eps_t = self.epsilon0 / max(1, self.t)  # decay by 1/t

        # explore vs exploit
        if np.random.rand() < eps_t:
            j = np.random.randint(self.k)  # explore
        else:
            # exploit the arm with the highest estimated mean
            est_means = [a.mean_est for a in self.arms]
            j = int(np.argmax(est_means))

        reward = self._sample_reward(self.arms[j].true_mean, precision=1.0)
        return j, reward

    def update(self, arm_index: int, reward: float):
        """Update the chosen arm’s running estimate with the observed reward.

        Args:
          arm_index: int: 
          reward: float: 

        Returns:

        """
        self.arms[arm_index].observe(reward)


    def experiment(self, n_trials: int = NumberOfTrials, algorithm_name: str = "EpsilonGreedy"):
        """Run epsilon-greedy for n_trials and log rows for CSV/plots.
        Stores a DataFrame at self.df with columns:
        t, Bandit, Reward, Regret, TrueMean, Algorithm

        Args:
          n_trials: int:  (Default value = NumberOfTrials)
          algorithm_name: str:  (Default value = "EpsilonGreedy")

        Returns:

        """
        for t in range(1, n_trials + 1):
            j, x = self.pull()
            self.update(j, x)
            regret = self.best_mean - self.arms[j].true_mean

            self.rows.append({
                "t": t,
                "Bandit": j,
                "Reward": x,
                "Regret": regret,
                "TrueMean": self.arms[j].true_mean,
                "Algorithm": algorithm_name
            })

        import pandas as pd  # safe import if top is missing
        self.df = pd.DataFrame(self.rows)

    def report(self, csv_path: str = "EpsilonGreedy_results.csv"):
        """Save CSV and log cumulative reward & regret via loguru.

        Args:
          csv_path: str:  (Default value = "EpsilonGreedy_results.csv")

        Returns:

        """
        if self.df is None:
            logger.error("EpsilonGreedy.report() called before experiment().")
            return

        self.df.to_csv(csv_path, index=False)
        total_reward = self.df["Reward"].sum()
        total_regret = self.df["Regret"].sum()
        logger.info(f"[Epsilon-Greedy] Cumulative reward = {total_reward:.2f}")
        logger.info(f"[Epsilon-Greedy] Cumulative regret = {total_regret:.2f}")

#--------------------------------------#

class ThompsonSampling(Bandit):
    """Thompson Sampling for Normal rewards with *known precision* (tau).
    Conjugate prior on each arm mean: Normal(mu0, v0).

    Args:

    Returns:

    """
    def __init__(self, p, tau: float = TS_KNOWN_PRECISION, mu0: float = TS_PRIOR_MEAN, v0: float = TS_PRIOR_VAR):
        """
        p   : list of true means for each arm (e.g., [1,2,3,4])
        tau : known precision of observation noise (1/sigma^2)
        mu0 : prior mean for arm mean
        v0  : prior variance for arm mean
        """
        self.true_means = list(p)
        self.k = len(self.true_means)

        self.tau = float(tau)
        self.mu0 = float(mu0)
        self.v0  = float(v0)

        # One ArmState per arm to keep running sums & sample means
        self.arms = [ArmState(m) for m in self.true_means]

        # For logging rows -> DataFrame
        self.rows = []
        self.df = None

        # For regret computation
        self.best_mean = max(self.true_means)

    def __repr__(self):
        return f"ThompsonSampling(k={self.k}, tau={self.tau}, mu0={self.mu0}, v0={self.v0})"

    def _sample_reward(self, mean: float) -> float:
        """Draw a Normal reward with known precision tau (σ = 1/√tau).

        Args:
          mean: float: 

        Returns:

        """
        sigma = (1.0 / self.tau) ** 0.5
        return np.random.normal(loc=mean, scale=sigma)

    def _posterior_params(self, arm: ArmState):
        """Normal–Normal conjugacy for known variance:
          v_n = 1 / (1/v0 + n * tau)
          mu_n = v_n * (mu0/v0 + tau * sum_x)
        Returns (mu_n, v_n)

        Args:
          arm: ArmState: 

        Returns:

        """
        denom = (1.0 / self.v0) + arm.n * self.tau
        v_n = 1.0 / denom
        mu_n = v_n * ((self.mu0 / self.v0) + (self.tau * arm.sum_x))
        return mu_n, v_n

    def pull(self):
        """Thompson step: sample a mean from each arm's posterior, pick the argmax,
        then observe a reward from the environment of that chosen arm.
        Returns: (arm_index, observed_reward)

        Args:

        Returns:

        """
        samples = []
        for arm in self.arms:
            mu_n, v_n = self._posterior_params(arm)
            samples.append(np.random.normal(loc=mu_n, scale=np.sqrt(v_n)))
        j = int(np.argmax(samples))
        reward = self._sample_reward(self.arms[j].true_mean)
        return j, reward

    def update(self, arm_index: int, reward: float):
        """Update sufficient statistics for the chosen arm.

        Args:
          arm_index: int: 
          reward: float: 

        Returns:

        """
        self.arms[arm_index].observe(reward)

    def experiment(self, n_trials: int = NumberOfTrials, algorithm_name: str = "ThompsonSampling"):
        """Run Thompson Sampling for n_trials and record results.
        Creates self.df with columns: t, Bandit, Reward, Regret, TrueMean, Algorithm

        Args:
          n_trials: int:  (Default value = NumberOfTrials)
          algorithm_name: str:  (Default value = "ThompsonSampling")

        Returns:

        """
        for t in range(1, n_trials + 1):
            j, x = self.pull()
            self.update(j, x)
            regret = self.best_mean - self.arms[j].true_mean

            self.rows.append({
                "t": t,
                "Bandit": j,
                "Reward": x,
                "Regret": regret,
                "TrueMean": self.arms[j].true_mean,
                "Algorithm": algorithm_name
            })

        import pandas as pd
        self.df = pd.DataFrame(self.rows)

    def report(self, csv_path: str = "ThompsonSampling_results.csv"):
        """Save CSV and log cumulative reward & regret.

        Args:
          csv_path: str:  (Default value = "ThompsonSampling_results.csv")

        Returns:

        """
        if self.df is None:
            logger.error("ThompsonSampling.report() called before experiment().")
            return

        self.df.to_csv(csv_path, index=False)
        total_reward = self.df["Reward"].sum()
        total_regret = self.df["Regret"].sum()
        logger.info(f"[Thompson Sampling] Cumulative reward = {total_reward:.2f}")
        logger.info(f"[Thompson Sampling] Cumulative regret = {total_regret:.2f}")





def comparison(df_eg, df_ts):
    """Extra side-by-side comparison:
    (1) Average reward over time
    (2) Cumulative regret
    (3) Arm selection share by algorithm

    Args:
      df_eg: 
      df_ts: 

    Returns:

    """
    import matplotlib.pyplot as plt

    # --- Line plots: avg reward + cumulative regret ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharex=False)

    # Average reward over time
    ax[0].plot(df_eg["t"], df_eg["Reward"].cumsum() / df_eg["t"], label="Epsilon-Greedy")
    ax[0].plot(df_ts["t"], df_ts["Reward"].cumsum() / df_ts["t"], label="Thompson Sampling")
    ax[0].set_title("Average Reward Over Time")
    ax[0].set_xlabel("Trial")
    ax[0].set_ylabel("Average Reward")
    ax[0].legend()

    # Cumulative regret
    ax[1].plot(df_eg["t"], df_eg["Regret"].cumsum(), label="Epsilon-Greedy")
    ax[1].plot(df_ts["t"], df_ts["Regret"].cumsum(), label="Thompson Sampling")
    ax[1].set_title("Cumulative Regret")
    ax[1].set_xlabel("Trial")
    ax[1].set_ylabel("Regret")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    # --- Bar chart: arm selection share ---
    sel_eg = df_eg["Bandit"].value_counts().sort_index()
    sel_ts = df_ts["Bandit"].value_counts().sort_index()
    arms = sorted(set(sel_eg.index).union(set(sel_ts.index)))

    eg_vals = [sel_eg.get(a, 0) for a in arms]
    ts_vals = [sel_ts.get(a, 0) for a in arms]

    width = 0.35
    x = np.arange(len(arms))

    plt.figure(figsize=(8, 5))
    plt.bar(x - width/2, eg_vals, width, label="Epsilon-Greedy")
    plt.bar(x + width/2, ts_vals, width, label="Thompson Sampling")
    plt.xticks(x, [f"Arm {a}" for a in arms])
    plt.ylabel("Selections (count)")
    plt.title("Arm Selection Share by Algorithm")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logger.info("Starting experiments...")

    # --- Run Epsilon-Greedy (εt = ε0/t) ---
    eg = EpsilonGreedy(Bandit_Reward, epsilon0=0.5)
    eg.experiment(n_trials=NumberOfTrials)
    eg.report(csv_path="EpsilonGreedy_results.csv")  # logs total reward & regret

    # --- Run Thompson Sampling (Normal, known variance) ---
    ts = ThompsonSampling(Bandit_Reward)
    ts.experiment(n_trials=NumberOfTrials)
    ts.report(csv_path="ThompsonSampling_results.csv")  # logs total reward & regret

    # --- Minimal CSV required by HW: {Bandit, Reward, Algorithm} ---
    df_min = pd.concat(
        [
            eg.df[["Bandit", "Reward"]].assign(Algorithm="EpsilonGreedy"),
            ts.df[["Bandit", "Reward"]].assign(Algorithm="ThompsonSampling"),
        ],
        ignore_index=True
    )
    df_min.to_csv("AllRewards_Min.csv", index=False)
    logger.info("Saved AllRewards_Min.csv with columns {Bandit, Reward, Algorithm}")

    # --- Plots ---
    vis = Visualization()
    vis.plot1(eg.df, title_prefix="Epsilon-Greedy")
    vis.plot1(ts.df, title_prefix="Thompson Sampling")
    vis.plot2(eg.df, ts.df)
    comparison(eg.df, ts.df)


    logger.info("Done.")





