# PACE: Project Idea Review

### 1. The Decision Problem and Why it is Interesting

**The Decision Problem:**
In competitive service selection, autonomous buyers repeatedly route requests to competing providers. These providers are shared, congestible resources: success probabilities decrease as concurrent load increases. Buyers must maximize their cumulative surplus subject to a strict, finite financial budget, paying a provider-specific cost per attempt. 

**Why it is Interesting:**
This environment features two interacting forces that make it a true Multi-Agent Systems problem: **temporal coupling** (spending now reduces future budget) and **strategic externality** (choosing a provider lowers others' current success probabilities). The core phenomenon is a continuous feedback loop: individual optimal decisions $\rightarrow$ collective congestion $\rightarrow$ changed rewards $\rightarrow$ changed future decisions.

To navigate this, buyers use a sliding-window contextual Thompson Sampling algorithm combined with Bandits with Knapsacks (BwK) style budget pacing. Crucially, buyers use a *lagged* public load signal as context. This introduces a fundamental identification problem: current success depends on *simultaneous* actions, making lagged load an intentionally imperfect predictive proxy. We investigate whether empirical adaptation and budget pacing remain beneficial despite this missing endogenous state.

***

### 2. The Agents, Objectives, and Game Flow

The environment is a repeated stochastic game with $K$ heterogeneous buyers and $N$ providers.

*   **Buyers ($k = 1, \dots, K$):** Each buyer $k$ has a private value $V_k$ and budget $B_k(t)$. In round $t$, they choose an action $a_k(t) \in \{0, 1, \dots, N\}$. 
    *   Action $0$ is "abstain." We set $p_0=0$ and $\tilde{q}_0=0$, meaning abstention always yields a score and cost of 0.
    *   Action $i \in \{1..N\}$ incurs a fixed price $p_i$. Payment is deducted upon attempt, regardless of success: $B_k(t+1) = B_k(t) - p_{a_k(t)}$ (with $p_0=0$ when abstaining).
    *   **Objective:** Each buyer $k$ seeks to maximize **Expected Individual Surplus**: $\max \mathbb{E}[U_k]$, where $U_k = \sum_{t=1}^T (V_k \cdot X_k(t) - p_{a_k(t)})$ and $X_k(t) \in \{0, 1\}$ is the Bernoulli success.
    *   **System Metric:** Aggregate Expected Buyer Surplus $W = \sum_k \mathbb{E}[U_k]$. *Note: Because buyers are decentralized and individually rational, they optimize only their local penalized score; aggregate surplus is solely an evaluation metric, not the agents' objective.*
*   **Providers ($i = 1, \dots, N$):** Passive resources with maximum reliability $q_i^{max}$ and capacity $c_i$. 
    *   Congestion rule: $q_i(t) = q_i^{max} \cdot \min(1, (c_i / n_i(t))^\gamma)$ for $n_i(t) > 0$, and $q_i(t) = q_i^{max}$ for $n_i(t) = 0$.

**Round Timing & Game Flow:**
```text
1. Observe history, budget B_k(t), and lagged load n(t-1)
      ↓
2. All buyers simultaneously select action a_k(t)
      ↓
3. Current aggregate load n_i(t) is formed
      ↓
4. Outcomes X_k(t) are sampled using congested reliability q_i(t)
      ↓
5. Payments p_i are deducted, budgets updated, and learning posteriors updated
```
*Note: After step 3, the current aggregate load $n_i(t)$ is **not** broadcast to agents; only the lagged load $n_i(t-1)$ remains public for the next decision step. We deliberately hide $n_i(t)$ to keep the learning problem strictly lag-based.*

***

### 3. What is Known and What is Unknown?

*   **Globally Known:** Provider count $N$, fixed prices $p_i$, the mathematical form of congestion, and lagged load $n_i(t-1)$. 
    *   *Information Leakage:* Because prices are public and assigned monotonically by quality, buyers infer the *ordinal ranking* of provider reliabilities. This partial information is explicitly incorporated into the learners' priors, though exact $q_i^{max}$ values remain unknown.
*   **Unknown:** Buyers do not know exact parameters ($q_i^{max}$, $c_i$, $\gamma$), nor the specific budgets or values of competitors. 

**Environment Parameters:**
| Parameter | Description / Distribution |
| :--- | :--- |
| $N, K, T$ | 5 Providers, 50 Buyers, $T=5,000$ rounds. |
| $V_k$ | $V_k \sim \text{Uniform}(4.0, 10.0)$. |
| $q_i^{max}, c_i$ | $q_i^{max} \sim \text{Uniform}(0.5, 1.0)$. $c_i \sim \text{Uniform}(5, 20)$. |
| Prices ($p_i$) | Providers sorted by $q_i^{max}$, prices assigned monotonically from $\$1.00$ to $\$3.00$. Average price $\bar{p} = 2.0$. |
| $\gamma$ | Fixed to $1.0$ for primary experiments. Sensitivity runs ($\gamma \in \{0.5, 2.0\}$) reported in Appendix. |
| Budget Tightness ($\tau$) | Target $\tau \in \{2.0 \text{ (Tight)}, 1.0 \text{ (Moderate)}, 0.5 \text{ (Slack)}\}$. To maintain buyer heterogeneity, initial budgets are sampled: $B_k(0) \sim \text{Uniform}(0.8, 1.2) \times \frac{T \cdot \bar{p}}{\tau}$. |
| Contention ($\rho$) | $\rho = K / \sum c_i$. Scaled to target values **Low (0.5)**, **Medium (1.0)**, **High (2.0)** after base $c_i$ generation. |

***

### 4. The Precise Computational Question

The central scientific question is: **When does budget pacing become valuable under endogenous congestion?** 
Specifically, we investigate:
1. Does load-aware contextual learning help?
2. Does sliding-window forgetting help?
3. Does pacing help?
4. Does the *full system's* overall surplus advantage over a vanilla baseline strengthen when contention and budget tightness increase simultaneously?

***

### 5. Algorithms and Solution Approach

We compare PACE against a matrix of references and independent learning baselines.

**1. Contextual TS Posterior & Sliding Window Mechanism**
Each buyer maintains a Beta posterior for every provider $i$ and deterministic load bucket $b$. Because $n_i(t) \in [0, 50]$ and $c_i \in [5, 20]$, we set a **bucket width of 5** (e.g., loads 0-4, 5-9) so context captures meaningful capacity saturation stages.
*   **Prior:** To exploit the price-quality rank leak ($p_i \in [1, 3]$), we initialize with an informed monotonic prior: $\alpha_{i,b} = p_i, \beta_{i,b} = 4 - p_i$.
*   **Update:** For an observation $X_k(t)$, the active bucket $b = \text{bucket}(n_i(t-1))$ updates: $\alpha_{i,b} \gets \alpha_{i,b} + X_k(t)$ and $\beta_{i,b} \gets \beta_{i,b} + (1-X_k(t))$. 
*   **Sliding Window:** A FIFO queue tracks observations. When an observation exceeds age $W$, its specific $X \in \{0,1\}$ is subtracted from the corresponding $\alpha$ or $\beta$ counter. Because old observations are explicitly decremented, an empty bucket's counters naturally revert to the prior, from which it samples seamlessly.
*   **Sampling:** $\tilde{q}_{i,b}(t) \sim \text{Beta}(\alpha_{i,b}, \beta_{i,b})$. The sampled success probability is $\tilde{q}_i(t) = \tilde{q}_{i, \text{bucket}(n_i(t-1))}(t)$.

**2. Budget Pacing and Feasibility**
To strictly prevent budget violations, we use a **Feasibility Filter**: $\mathcal{A}_k(t) = \{0\} \cup \{i : p_i \le B_k(t)\}$. 
Action selection relies on a **Pacing Multiplier** $\phi_k(t)$, penalizing expensive actions:
$$a_k(t) = \arg\max_{i \in \mathcal{A}_k(t)} \left[ V_k \tilde{q}_{i}(t) - p_i(1 + \phi_k(t)) \right]$$
$\phi_k$ is a heuristic motivated by the sub-gradient of the per-round budget constraint, treating the multiplier as a tunable pacing factor. It updates via:
$$\phi_k(t+1) = \max\left(0, \phi_k(t) + \eta \left( p_{a_k(t)} - \frac{B_k(t)}{T - t + 1} \right) \right)$$

**3. Baselines & Benchmarks**
*   **Vanilla Cumulative TS + Feasibility:** A standard bandit baseline that ignores contexts, pacing, and windows. Tests whether our architectural complexity improves upon standard out-of-the-box learning.
*   **Naïve Random:** Uniformly samples $\mathcal{A}_k(t)$. Anchors the uninformed baseline.
*   **Centralized Greedy Heuristic:** A naive centralized reference. In each round $t$, it greedily assigns buyers to providers in descending order of $V_k$ to maximize their *marginal* immediate expected surplus given prior assignments, subject to remaining budgets.
*   **Oracle Centralized Planner:** Approximates the true combinatorial optimum using a receding-horizon Mixed-Integer Program (MIP), granting it full true-parameter knowledge without requiring solving the non-convex $T=5000$ horizon instantly. Serves as a privileged reference benchmark. [^1]

[^1]: A Stage-Game Nash Benchmark is treated purely as a static strategic reference in the appendix, as it ignores intertemporal budget dynamics.

***

### 6. Evaluation Methodology

A custom Python simulator evaluates the environment across a $3 \times 3$ grid of Contention ($\rho$) and Tightness ($\tau$).
*   **Hyperparameters:** $W = 200$, $\eta = 0.05$. 
*   **Seed Generation:** To isolate environmental scaling, 50 base environments are sampled (10 validation, 40 holdout). These exact base environments are then structurally scaled across the $3 \times 3$ grid. The same seeds are reused across all algorithms to guarantee a fair **common-random-numbers** (CRN) evaluation.

**Metrics:**
1.  *Aggregate Expected Buyer Surplus ($W$)*.
2.  *Coefficient of Variation (CV):* $\text{CV}_i = \sigma_i / \mu_i$ computed per provider over 5,000 rounds. (If $\mu_i=0$, $\text{CV}_i=0$).
3.  *Cross-Provider Herding (HHI):* $HHI_t = \sum_i (n_i(t) / \sum_j n_j(t))^2$. (If total load is 0, HHI = 0).

**A. Component-Interaction Study (Algorithm Synergy)**
To rigorously measure interaction among pacing ($P$), context ($C$), and sliding windows ($S$), we test $2^3$ factorial configurations (e.g., $000$ is Vanilla TS, $111$ is PACE). For each environmental cell $c$, we fit a factorial ANOVA on the seed-level realized surplus $W_{s,c}$, incorporating a seed fixed effect $\delta_s$ to properly exploit the paired CRN design:
$$W_{s,c} = \theta_0 + \theta_P P + \theta_C C + \theta_S S + \theta_{PC} PC + \theta_{PS} PS + \theta_{CS} CS + \theta_{PCS} PCS + \delta_s + \varepsilon_{s,c}$$

**B. Environmental-Interaction Study (Contention $\times$ Budget)**
To test whether the full system's advantage over the baseline strengthens when *both* contention and budget tightness are high, we pool across the 9 cells and analyze relative performance $\Delta W_{s,c} = W^{\text{PACE}}_{s,c} - W^{\text{Vanilla TS}}_{s,c}$. Because the 50 base environments are structurally scaled across the grid, the global seed fixed effect $\delta_s$ mathematically holds:
$$\Delta W_{s,c} = \beta_0 + \beta_1 \rho_c + \beta_2 \tau_c + \beta_3 (\rho_c \tau_c) + \delta_s + \varepsilon_{s,c}$$

***

### 7. Success Criteria and Limitations

*   **Success Criteria:** The project succeeds if the statistical analysis shows that each component (pacing, context, sliding-window) is *beneficial* in at least one environmental regime (evaluated at a Bonferroni-corrected $\alpha = 0.05/8$ to conservatively control for multiple comparisons across configurations), and the environmental regression identifies a **positive interaction** ($\beta_3 > 0$) between contention and budget tightness. 
    *   *Note:* A null or negative $\beta_3$ would be a meaningful negative result, implying that pacing mechanisms fail to scale or actively collapse under high systemic stress.
*   **Limitations:** The Oracle Centralized Planner is an unreachable benchmark for decentralized agents. Because the lagged context is inherently flawed in a simultaneous-action game, no theoretical optimality bounds can be guaranteed for the learners. The Centralized Greedy heuristic is short-sighted and may underperform learners in heavily constrained regimes.

***

### 8. Why this Project is Unique

This project formally resides at the intersection of **Congestion Games** (e.g., Rosenthal 1973) and **Bandits with Knapsacks (BwK)** (e.g., Agrawal and Devanur 2014). While standard Contextual BwK optimal-pacing literature assumes contexts are strictly exogenous, and multi-agent congestion literature typically assumes infinite horizons without hard stopping budgets, our environment merges them. We formulate a synthetic, highly-constrained setting where the context (lagged load) is *endogenous* and adversarial. Rather than proposing a purely novel mathematical algorithm, this project's uniqueness lies in empirically investigating whether combining known mechanisms (Contextual BwK pacing, Thompson Sampling, and Sliding Windows) can robustly survive policy-induced non-stationarity under strict financial exhaustion.
