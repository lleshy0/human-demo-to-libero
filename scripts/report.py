"""Results tables in markdown, one row per run, read from runs/eval_*/results.json."""
import json
import os
import sys

from splits import ALIGN, ROBOT_ONLY, TARGET

SHORT = {
    "put_the_bowl_on_top_of_the_cabinet": "bowl on cabinet", "put_the_wine_bottle_on_top_of_the_cabinet": "bottle on cabinet",
    "push_the_plate_to_the_front_of_the_stove": "push plate", "open_the_middle_drawer_of_the_cabinet": "middle drawer",
    "turn_on_the_stove": "stove knob", "put_the_wine_bottle_on_the_rack": "bottle on rack",
    "open_the_top_drawer_and_put_the_bowl_inside": "drawer+bowl", "put_the_bowl_on_the_plate": "**bowl on plate**",
    "put_the_bowl_on_the_stove": "**bowl on stove**", "put_the_cream_cheese_in_the_bowl": "**cheese in bowl**",
}
ORDER = list(ALIGN) + list(ROBOT_ONLY) + list(TARGET)  # TARGET last, in bold


def row(name):
    eps = json.load(open(f"runs/eval_{name}/results.json"))["episodes"]
    return {t: f"{sum(e['success'] for e in eps if e['task'] == t)}/{sum(e['task'] == t for e in eps)}"
            for t in ORDER if any(e["task"] == t for e in eps)}


if __name__ == "__main__":
    rows = {n: row(n) for n in sys.argv[1:] if os.path.exists(f"runs/eval_{n}/results.json")}
    tasks = [t for t in ORDER if any(t in c for c in rows.values())]
    print("| run | " + " | ".join(SHORT[t] for t in tasks) + " |")
    print("|---|" + "---|" * len(tasks))
    for n, cells in rows.items():
        print(f"| {n} | " + " | ".join(cells.get(t, "") for t in tasks) + " |")
