import math
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(page_title="Hospital Vaccination Impact Simulation", layout="wide")


@dataclass
class Params:
    population: int
    initial_infected: int
    beta: float
    recovery_rate: float
    days: int
    contacts_per_day: int
    vaccine_coverage: float
    vaccine_efficacy: float
    intervention_day: int
    daily_vaccination_rate: float
    topology: str
    seed: int


def build_graph(n: int, topology: str, contacts_per_day: int, seed: int):
    avg_degree = max(2, min(n - 1, contacts_per_day * 2))
    if topology == "Random Contact Network":
        p = min(1.0, avg_degree / max(1, n - 1))
        return nx.erdos_renyi_graph(n, p, seed=seed)
    if topology == "Scale-Free Hospital Hubs":
        m = max(1, min((avg_degree // 2), n - 1))
        return nx.barabasi_albert_graph(n, m, seed=seed)
    k = max(2, min(avg_degree, n - 1))
    if k % 2 == 1:
        k += 1
    k = min(k, n - 1 if (n - 1) % 2 == 0 else n - 2)
    return nx.watts_strogatz_graph(n, max(2, k), 0.15, seed=seed)


def run_simulation(params: Params):
    rng = random.Random(params.seed)
    np.random.seed(params.seed)

    G = build_graph(params.population, params.topology, params.contacts_per_day, params.seed)

    SUSCEPTIBLE = "S"
    VACCINATED = "V"
    INFECTED = "I"
    RECOVERED = "R"

    state = {node: SUSCEPTIBLE for node in G.nodes()}
    infected_days = {node: 0 for node in G.nodes()}

    # Initial vaccination before outbreak
    initial_vaccinated_count = int(params.population * params.vaccine_coverage)
    vacc_candidates = list(G.nodes())
    rng.shuffle(vacc_candidates)
    for node in vacc_candidates[:initial_vaccinated_count]:
        state[node] = VACCINATED

    # Initial infected from non-vaccinated if possible
    infect_candidates = [n for n in G.nodes() if state[n] != VACCINATED]
    if len(infect_candidates) < params.initial_infected:
        infect_candidates = list(G.nodes())
    rng.shuffle(infect_candidates)
    for node in infect_candidates[:params.initial_infected]:
        state[node] = INFECTED
        infected_days[node] = 1

    history = []

    for day in range(params.days + 1):
        counts = {
            "day": day,
            "Susceptible": sum(1 for s in state.values() if s == SUSCEPTIBLE),
            "Vaccinated": sum(1 for s in state.values() if s == VACCINATED),
            "Infected": sum(1 for s in state.values() if s == INFECTED),
            "Recovered": sum(1 for s in state.values() if s == RECOVERED),
        }
        history.append(counts)

        if day == params.days:
            break

        # Intervention vaccination campaign
        if day >= params.intervention_day:
            susceptible_nodes = [n for n in G.nodes() if state[n] == SUSCEPTIBLE]
            rng.shuffle(susceptible_nodes)
            vaccinate_today = int(len(susceptible_nodes) * params.daily_vaccination_rate)
            for node in susceptible_nodes[:vaccinate_today]:
                state[node] = VACCINATED

        new_infections = set()
        recoveries = set()

        infected_nodes = [n for n in G.nodes() if state[n] == INFECTED]
        for node in infected_nodes:
            neighbors = list(G.neighbors(node))
            rng.shuffle(neighbors)
            sampled_neighbors = neighbors[: min(len(neighbors), params.contacts_per_day)]
            for nb in sampled_neighbors:
                if state[nb] == SUSCEPTIBLE:
                    if rng.random() < params.beta:
                        new_infections.add(nb)
                elif state[nb] == VACCINATED:
                    reduced_beta = params.beta * (1 - params.vaccine_efficacy)
                    if rng.random() < reduced_beta:
                        new_infections.add(nb)

            if rng.random() < params.recovery_rate:
                recoveries.add(node)

        for node in new_infections:
            state[node] = INFECTED
            infected_days[node] = infected_days.get(node, 0) + 1

        for node in recoveries:
            state[node] = RECOVERED

    df = pd.DataFrame(history)
    peak_infected = int(df["Infected"].max())
    peak_day = int(df.loc[df["Infected"].idxmax(), "day"])
    total_ever_infected = int(df.iloc[-1]["Recovered"] + df.iloc[-1]["Infected"])

    R0_est = (params.beta * params.contacts_per_day) / max(params.recovery_rate, 1e-9)
    herd_threshold = max(0.0, 1 - 1 / R0_est) if R0_est > 1 else 0.0
    effective_protection = params.vaccine_coverage * params.vaccine_efficacy
    herd_status = effective_protection >= herd_threshold

    return {
        "graph": G,
        "df": df,
        "peak_infected": peak_infected,
        "peak_day": peak_day,
        "total_ever_infected": total_ever_infected,
        "R0": R0_est,
        "herd_threshold": herd_threshold,
        "effective_protection": effective_protection,
        "herd_status": herd_status,
        "state": state,
    }


st.title("🏥 Hospital Vaccination Impact Simulation")
st.caption("SIR-inspired outbreak model with vaccination, intervention strategy, and herd immunity analysis.")

with st.sidebar:
    st.header("Simulation Controls")
    population = st.slider("Patients / individuals", 50, 600, 180, 10)
    initial_infected = st.slider("Initially infected", 1, 20, 3)
    beta = st.slider("Transmission probability", 0.01, 1.0, 0.18, 0.01)
    recovery_rate = st.slider("Recovery rate", 0.01, 1.0, 0.10, 0.01)
    days = st.slider("Simulation days", 15, 180, 60, 5)
    contacts_per_day = st.slider("Daily risky contacts", 1, 20, 6)
    vaccine_coverage = st.slider("Initial vaccination coverage", 0.0, 1.0, 0.35, 0.01)
    vaccine_efficacy = st.slider("Vaccine efficacy", 0.0, 1.0, 0.75, 0.01)
    intervention_day = st.slider("Vaccination campaign starts on day", 0, 60, 10)
    daily_vaccination_rate = st.slider("Daily vaccination rate after intervention", 0.0, 0.5, 0.05, 0.01)
    topology = st.selectbox(
        "Hospital interaction pattern",
        ["Random Contact Network", "Scale-Free Hospital Hubs", "Small-World Wards"],
    )
    seed = st.number_input("Random seed", 0, 9999, 42)

params = Params(
    population=population,
    initial_infected=initial_infected,
    beta=beta,
    recovery_rate=recovery_rate,
    days=days,
    contacts_per_day=contacts_per_day,
    vaccine_coverage=vaccine_coverage,
    vaccine_efficacy=vaccine_efficacy,
    intervention_day=intervention_day,
    daily_vaccination_rate=daily_vaccination_rate,
    topology=topology,
    seed=seed,
)

results = run_simulation(params)
df = results["df"]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak infected", results["peak_infected"])
m2.metric("Peak day", results["peak_day"])
m3.metric("Estimated R₀", f"{results['R0']:.2f}")
m4.metric("Total ever infected", results["total_ever_infected"])

c1, c2 = st.columns([1.5, 1])

with c1:
    st.subheader("Compartment Trends")
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["Susceptible", "Vaccinated", "Infected", "Recovered"]:
        ax.plot(df["day"], df[col], label=col)
    ax.set_xlabel("Day")
    ax.set_ylabel("People")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

with c2:
    st.subheader("Herd Immunity Analysis")
    herd_df = pd.DataFrame(
        {
            "Metric": [
                "Effective protection",
                "Herd immunity threshold",
                "Vaccination coverage",
                "Vaccine efficacy",
            ],
            "Value": [
                f"{results['effective_protection']*100:.1f}%",
                f"{results['herd_threshold']*100:.1f}%",
                f"{vaccine_coverage*100:.1f}%",
                f"{vaccine_efficacy*100:.1f}%",
            ],
        }
    )
    st.table(herd_df)
    if results["herd_status"]:
        st.success("Estimated effective protection meets or exceeds the herd immunity threshold.")
    else:
        st.warning("Estimated effective protection is below the herd immunity threshold.")

st.subheader("Intervention Insights")
insight_cols = st.columns(3)
insight_cols[0].info(
    f"Vaccination campaign starts on day {intervention_day} and vaccinates about {daily_vaccination_rate*100:.1f}% of remaining susceptible patients per day."
)
insight_cols[1].info(
    f"Initial vaccination coverage protects about {vaccine_coverage*100:.1f}% of the population before the outbreak starts."
)
insight_cols[2].info(
    f"With vaccine efficacy at {vaccine_efficacy*100:.1f}%, breakthrough infections are reduced but still possible."
)

st.subheader("Final Network Snapshot")
G = results["graph"]
state = results["state"]
color_map = {
    "S": "#9ecae1",
    "V": "#74c476",
    "I": "#fb6a4a",
    "R": "#bcbddc",
}
fig2, ax2 = plt.subplots(figsize=(10, 7))
pos = nx.spring_layout(G, seed=seed, k=1 / math.sqrt(max(1, population)))
node_colors = [color_map[state[n]] for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=80, node_color=node_colors, ax=ax2)
nx.draw_networkx_edges(G, pos, alpha=0.15, ax=ax2)
ax2.set_axis_off()
st.pyplot(fig2)

with st.expander("Show simulation data table"):
    st.dataframe(df, use_container_width=True)

st.markdown("---")
st.markdown(
    "**Model states:** Susceptible patients, Vaccinated, Infected, and Recovered. "
    "This dashboard is suitable for academic demos, healthcare analytics prototypes, and intervention strategy comparison."
)
