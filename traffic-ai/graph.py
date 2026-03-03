"""
graph.py — Generate traffic density bar chart using matplotlib.
Saves to static/graphs/density.png and returns the path.
"""

import os
from typing import Dict

def generate_density_graph(
    lane_counts: Dict[int, int],
    output_path: str,
) -> str:
    """
    Creates a styled bar chart of vehicle counts per lane.

    Args:
        lane_counts  : {lane_id: vehicle_count}
        output_path  : full path to save the PNG

    Returns:
        output_path on success, "" on failure
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        lanes  = [f"Lane {i+1}" for i in sorted(lane_counts.keys())]
        counts = [lane_counts[i] for i in sorted(lane_counts.keys())]

        # Colour bars by density
        max_c  = max(counts) if counts else 1
        colors = []
        for c in counts:
            ratio = c / max(max_c, 1)
            if ratio > 0.66:
                colors.append("#ff4444")
            elif ratio > 0.33:
                colors.append("#ffcc00")
            else:
                colors.append("#00e676")

        fig, ax = plt.subplots(figsize=(7, 3.6))
        fig.patch.set_facecolor("#0d1b2e")
        ax.set_facecolor("#0d1b2e")

        bars = ax.bar(lanes, counts, color=colors, width=0.5,
                      edgecolor="#ffffff22", linewidth=0.8)

        # Value labels on bars
        for bar, val in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val),
                ha="center", va="bottom",
                color="#e0e0e0", fontsize=11, fontweight="bold",
            )

        ax.set_title("Vehicle Density per Lane", color="#e0e0e0",
                     fontsize=14, fontweight="bold", pad=12)
        ax.set_ylabel("Vehicle Count", color="#a0aab4", fontsize=10)
        ax.set_xlabel("Lane", color="#a0aab4", fontsize=10)
        ax.tick_params(colors="#a0aab4", labelsize=10)
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a3f5f")

        # Legend
        legend_items = [
            mpatches.Patch(color="#00e676", label="Low density"),
            mpatches.Patch(color="#ffcc00", label="Medium density"),
            mpatches.Patch(color="#ff4444", label="High density"),
        ]
        ax.legend(handles=legend_items, facecolor="#0d1b2e",
                  edgecolor="#2a3f5f", labelcolor="#c8d6e5", fontsize=9)

        ax.set_ylim(0, max(max_c * 1.3, 5))
        ax.yaxis.grid(True, color="#1e3a5f", linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        return output_path

    except Exception as e:
        print(f"⚠  Graph generation failed: {e}")
        return ""
