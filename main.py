import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np


class MoralNetworkSimulation:
    def __init__(self, n_nodes=12, p=0.1, n_steps=25, seed=42):
        np.random.seed(seed)
        self.G = nx.watts_strogatz_graph(n_nodes, k=3, p=0.1, seed=seed)
        self.pos = nx.circular_layout(self.G)
        self.n_steps = n_steps
        self.snapshots = []
        self.stats = []
        # initial moral states (-1 evil, 1 good)
        self.states = {n: np.random.uniform(-1, 1) for n in self.G.nodes}
        self.curr_t = 0

    def evolve(self):
        """Simulate moral dynamics over time."""
        for t in range(self.curr_t, self.n_steps):
            new_states = {}
            for node in self.G.nodes:
                neighbors = list(self.G.neighbors(node))
                if neighbors:
                    neighbor_mean = np.mean([self.states[n] for n in neighbors])
                else:
                    neighbor_mean = 0
                # update with local influence and slight noise
                new_states[node] = (
                    0.8 * self.states[node] + 0.2 * neighbor_mean + np.random.normal(0, 0.05)
                )
                new_states[node] = np.clip(new_states[node], -1, 1)

            # occasional moral conversions
            if np.random.rand() < 0.2:
                flip = np.random.choice(list(self.G.nodes))
                new_states[flip] *= -1

            self.states = new_states

            # store state in node attributes for snapshot
            for n, s in self.states.items():
                self.G.nodes[n]["state"] = s

            # collect statistics
            values = np.array(list(self.states.values()))
            avg_good = np.mean(values)
            polarity = np.mean(np.abs(values))
            self.stats.append({"t": t, "avg_good": avg_good, "polarity": polarity})

            # snapshot
            self.snapshots.append(self.G.copy())
        print(self.stats)

    def _make_graph_frame(self, g):
        """Create network visualization for a given snapshot."""
        edge_x, edge_y = [], []
        for u, v in g.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [self.pos[k][0] for k in g.nodes]
        node_y = [self.pos[k][1] for k in g.nodes]
        node_color = [g.nodes[k]["state"] for k in g.nodes]
        node_size = [abs(g.nodes[k]["state"]) * 20 + 10 for k in g.nodes]

        return [
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines",
                line=dict(width=1, color="gray"), hoverinfo="none"
            ),
            go.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                text=[f"{k}" for k in g.nodes],
                textposition="top center",
                marker=dict(
                    size=node_size,
                    color=node_color,
                    colorscale=[(0, "red"), (0.5, "lightgray"), (1, "blue")],
                    cmin=-1, cmax=1,
                    colorbar=dict(title="Moral<br>State")
                )
            ),
        ]

    def make_figure(self):
        """Create full interactive animation with moral dynamics."""
        t = [s["t"] for s in self.stats]
        avg_good = [s["avg_good"] for s in self.stats]
        polarity = [s["polarity"] for s in self.stats]
        y_min, y_max = -1, 1  # for cursor

        # Create subplots: network + stats panel
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.7, 0.3],
            subplot_titles=("Moral Network Evolution", "Moral Metrics"),
            specs=[[{"type": "scatter"}, {"type": "xy"}]]
        )

        # --- Initial network traces ---
        g0 = self.snapshots[0]
        graph_traces = self._make_graph_frame(g0)
        for tr in graph_traces:
            fig.add_trace(tr, row=1, col=1)

        # --- Static stats traces (added once, outside frames) ---
        avg_trace = go.Scatter(
            x=t, y=avg_good, mode="lines+markers",
            name="Avg Goodness", line=dict(color="blue")
        )
        pol_trace = go.Scatter(
            x=t, y=polarity, mode="lines+markers",
            name="Polarity", line=dict(color="orange")
        )
        fig.add_trace(avg_trace, row=1, col=2)
        fig.add_trace(pol_trace, row=1, col=2)

        # --- Moving cursor ---
        cursor = go.Scatter(
            x=[t[0], t[0]], y=[y_min, y_max],
            mode="lines", line=dict(color="red", dash="dash"), name="Time Cursor"
        )
        fig.add_trace(cursor, row=1, col=2)

        # --- Frames ---
        frames = []
        for i, g in enumerate(self.snapshots):
            print(i)
            graph_traces = self._make_graph_frame(g)
            f_cursor = go.Scatter(
                x=[t[i], t[i]], y=[y_min, y_max],
                mode="lines", line=dict(color="red", dash="dash"),
                showlegend=False
            )
            frames.append(go.Frame(
                data=graph_traces + [f_cursor],
                name=str(i)
            ))

        fig.update(frames=frames)

        # --- Layout & animation buttons ---
        fig.update_layout(
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                     "label": "▶️ Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                     "label": "⏸ Pause", "method": "animate"}
                ],
                "direction": "left",
                "x": 0.1, "xanchor": "right",
                "y": 1.15, "yanchor": "top"
            }],
            height=600,
            showlegend=True,
            margin=dict(l=20, r=20, t=50, b=20)
        )

        # --- Axes adjustments ---
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)
        fig.update_xaxes(title="Time", row=1, col=2)
        fig.update_yaxes(title="Moral Metric", row=1, col=2)

        # Keep aspect ratio for network
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)

        return fig

    def save(self, filename):
        """Save simulation state to a file."""
        data = {
            "G": self.G,
            "states": self.states,
            "snapshots": self.snapshots,
            "stats": self.stats,
            "pos": self.pos,
            "n_steps": self.n_steps,
            "curr_t": self.curr_t,
        }
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Simulation saved to {filename}")

    def load(self, filename, continue_for=250):
        """Load simulation state from a file."""
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.G = data["G"]
        self.states = data["states"]
        self.snapshots = data["snapshots"]
        self.stats = data["stats"]
        self.pos = data["pos"]
        self.n_steps = data["n_steps"]
        self.curr_t = data["curr_t"]
        self.n_steps += continue_for
        print(f"Simulation loaded from {filename}, {self.curr_t}")


# --- Example usage ---
sim = MoralNetworkSimulation(n_nodes=150, p=0.25, n_steps=250)
sim.load("moral_dynamics.pkl")
sim.evolve()
sim.save("moral_dynamics1.pkl")
fig = sim.make_figure()
fig.show()

# Optionally save
fig.write_html("moral_network_simulation.html", include_plotlyjs="cdn")
