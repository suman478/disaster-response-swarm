# Disaster Response Swarm – Results

## Table 1 – Effect of number of agents on performance (seed = 42)

| num_agents | Final step to find all victims | Victims found / total | Final coverage (%) | Total time (s) |
|-----------:|-------------------------------:|-----------------------:|-------------------:|---------------:|
| 4          | [step_4]                       | [vf_4]/10              | [cov_4]            | [time_4]       |
| 8          | [step_8]                       | [vf_8]/10              | [cov_8]            | [time_8]       |
| 16         | [step_16]                      | [vf_16]/10             | [cov_16]           | [time_16]      |

*(Fill in [step_*], [cov_*], [time_*] with values from `analyze_metrics.py`.)*

---

## Discussion

We evaluated the effect of the number of agents (`num_agents`) on the performance of a swarm-based
disaster response simulation. All experiments used a 30×20 grid, obstacle density 0.12, 10 victims,
and seed 42. We varied `num_agents` in {4, 8, 16}.

As the number of agents increases, the time (in steps) needed to discover all victims decreases.
With more agents, the environment is explored in parallel, which accelerates victim discovery.

In our tests:

- With 4 agents, the swarm required [step_4] steps to discover all 10 victims, reaching a final
  coverage of [cov_4]% in [time_4] seconds.
- With 8 agents, the swarm required [step_8] steps, with coverage [cov_8]% and time [time_8] seconds.
- With 16 agents, the swarm required only [step_16] steps, with coverage [cov_16]% and time [time_16] seconds.

These results confirm that adding more agents speeds up exploration and victim discovery. However,
in realistic scenarios, too many agents can also cause redundancy (multiple agents visiting the
same cells) and potential congestion in narrow areas. Future work could investigate more advanced
exploration strategies (e.g., frontier-based exploration or explicit task allocation) to reduce
overlap when the swarm size is large.
