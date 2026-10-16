"""The task split. Every robot training set is checked here, so TARGET robot data only enters diagnostic runs.
Run this file to check the split against LIBERO's task list.
"""
SUITE = "libero_goal"
TARGET = ("put_the_bowl_on_the_plate", "put_the_bowl_on_the_stove", "put_the_cream_cheese_in_the_bowl")
ALIGN = ("put_the_bowl_on_top_of_the_cabinet", "put_the_wine_bottle_on_top_of_the_cabinet")
ROBOT_ONLY = ("push_the_plate_to_the_front_of_the_stove", "open_the_middle_drawer_of_the_cabinet", "turn_on_the_stove",
              "put_the_wine_bottle_on_the_rack", "open_the_top_drawer_and_put_the_bowl_inside")
HUMAN = TARGET + ALIGN


def libero90_tasks():
    """LIBERO-90 tasks allowed as extra robot data (diagnostics only). Drops every task that puts a TARGET
    object on its TARGET destination (bowl+plate, bowl+stove, cream cheese+bowl)."""
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    leaks = lambda t: ("bowl" in t and ("plate" in t or "stove" in t)) or ("cream_cheese" in t and "bowl" in t)
    return tuple(t for t in libero_task_map["libero_90"] if not leaks(t))


def robot_tasks(diagnostic=False):
    """Tasks whose robot demos a run may train on."""
    return ALIGN + ROBOT_ONLY + (TARGET if diagnostic else ())


def check_robot(tasks, diagnostic=False):
    """Fail if TARGET robot data is in a non-diagnostic training set."""
    leak = set(tasks) & set(TARGET)
    assert diagnostic or not leak, f"TARGET robot data in a non-diagnostic run: {sorted(leak)}"
    return tasks


if __name__ == "__main__":
    from libero.libero.benchmark.libero_suite_task_map import libero_task_map

    assert sorted(TARGET + ALIGN + ROBOT_ONLY) == sorted(libero_task_map[SUITE]), "split must cover LIBERO-Goal exactly once"
    check_robot(robot_tasks())
    try:
        check_robot(robot_tasks(diagnostic=True))
        raise SystemExit("leak check failed to fire")
    except AssertionError:
        pass
    l90 = libero90_tasks()
    assert "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate" not in l90 and "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove" in l90
    print("libero_90 tasks kept:", len(l90))
    print("split ok:", {k: len(v) for k, v in dict(TARGET=TARGET, ALIGN=ALIGN, ROBOT_ONLY=ROBOT_ONLY).items()})
