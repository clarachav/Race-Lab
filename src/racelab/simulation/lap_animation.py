"""Animation d'un tour F1 2025 à Spa + fenêtre dédiée aux Scoreboards."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider

from racelab.models.tires import Tire
from racelab.models.track import (
    Segment,
    build_track,
    load_geojson_xy,
    track_from_xy,
    track_xy,
)
from racelab.models.vehicle import Vehicle
from racelab.simulation.lap_time import simulate_flying_lap

CIRCUIT = "spa"    # "spa" ou "oval"
REPO_ROOT = Path(__file__).resolve().parents[3]
SPA_FILE = REPO_ROOT / "data" / "raw" / "spa.geojson"
FPS = 30
SPEEDUP = 3.0      # 1 = temps réel, 3 = trois fois plus vite
TRAIL_FRAMES = 18
GAP_WINDOW_S = 10.0

# Paramètres d'estimation de sortie des stands
PIT_ENTRY_FRAC = 0.88
PIT_EXIT_FRAC = 0.04
PIT_LOSS_S = 20.0
PIT_TARGET_CODE = "NOR"

# Grille F1 2025 : (code, equipe, couleur, pace_factor)
# Calibré sur la saison 2025 avec victoire de NOR à la fin.
GRID_2025 = [
    ("NOR", "McLaren",      "#FF8000", 1.045),
    ("VER", "Red Bull",     "#3671C6", 1.041),
    ("PIA", "McLaren",      "#FFB366", 1.037),
    ("RUS", "Mercedes",     "#27F4D2", 1.029),
    ("LEC", "Ferrari",      "#E8002D", 1.023),
    ("HAM", "Ferrari",      "#FF5A79", 1.017),
    ("ANT", "Mercedes",     "#8CF9E7", 1.014),
    ("ALB", "Williams",     "#64C4FF", 1.006),
    ("SAI", "Williams",     "#1E88E5", 1.004),
    ("ALO", "Aston Martin", "#229971", 1.001),
    ("HUL", "Kick Sauber",  "#52E252", 0.998),
    ("HAD", "Racing Bulls", "#6692FF", 0.997),
    ("BEA", "Haas",         "#E1E4E8", 0.993),
    ("LAW", "Racing Bulls", "#A3BDFF", 0.991),
    ("OCO", "Haas",         "#B6BABD", 0.990),
    ("STR", "Aston Martin", "#66C2A3", 0.987),
    ("TSU", "Red Bull",     "#6C9EE8", 0.986),
    ("GAS", "Alpine",       "#0093CC", 0.981),
    ("BOR", "Kick Sauber",  "#99F099", 0.978),
    ("COL", "Alpine",       "#5CC4EC", 0.974),
]


def build_oval():
    r = 50
    return build_track([
        Segment(600),
        Segment(math.pi * r, radius=r),
        Segment(600),
        Segment(math.pi * r, radius=r),
    ])


def load_circuit():
    if CIRCUIT == "oval":
        return build_oval()
    x, y = load_geojson_xy(SPA_FILE)
    return track_from_xy(x, y, ds=5.0, smooth_m=50.0, closed=True)


def compute_sectors(t_car, track):
    """Calcule les temps de secteurs S1, S2, S3 et le temps total du tour."""
    total_len = float(track.length)
    s1_end = total_len / 3.0
    s2_end = 2.0 * total_len / 3.0

    t_s1 = float(np.interp(s1_end, track.s, t_car))
    t_s2 = float(np.interp(s2_end, track.s, t_car)) - t_s1
    t_s3 = float(t_car[-1]) - (t_s1 + t_s2)
    return t_s1, t_s2, t_s3, float(t_car[-1])


def build_car_profiles(base_result, track, x, y):
    """Calcule position, vitesse, distance, secteurs et chronologie de chaque voiture."""
    base_speed = np.asarray(base_result.speed, dtype=float)
    base_dt = 2.0 * track.ds / (base_speed[:-1] + base_speed[1:])
    base_t = np.concatenate(([0.0], np.cumsum(base_dt)))
    base_lap_time = float(base_t[-1])

    fastest_pace = max(item[3] for item in GRID_2025)
    winner_lap_time = base_lap_time / fastest_pace
    times = np.arange(0.0, winner_lap_time + (SPEEDUP / FPS), SPEEDUP / FPS)

    cars = []
    for code, team, color, pace in GRID_2025:
        speed = base_speed * pace
        v_kmh = speed * 3.6
        dt = 2.0 * track.ds / (speed[:-1] + speed[1:])
        t_car = np.concatenate(([0.0], np.cumsum(dt)))
        s1, s2, s3, lap_time = compute_sectors(t_car, track)

        t_query = np.minimum(times, lap_time)
        xs = np.interp(t_query, t_car, x)
        ys = np.interp(t_query, t_car, y)
        ss = np.interp(t_query, t_car, track.s)
        vs = np.interp(t_query, t_car, v_kmh)

        cars.append({
            "code": code,
            "team": team,
            "color": color,
            "pace": pace,
            "lap_time": lap_time,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "v_max_kmh": float(np.max(v_kmh)),
            "t_car": t_car,
            "xs": xs,
            "ys": ys,
            "ss": ss,
            "vs": vs,
        })

    cars.sort(key=lambda c: c["lap_time"])
    return times, cars


def estimate_pit_pos_for_car(car_idx, cars, track, frame_idx, pit_loss_s):
    """Estime la position et l'écart à la sortie des stands pour la voiture car_idx."""
    pit_entry_s = track.length * PIT_ENTRY_FRAC
    pit_exit_s = track.length * PIT_EXIT_FRAC

    target_car = cars[car_idx]
    s_target = float(target_car["ss"][frame_idx])
    v_target_ms = max(25.0, float(target_car["vs"][frame_idx]) / 3.6)
    avg_v_target = max(30.0, float(target_car["vs"].mean()) / 3.6)

    if s_target <= pit_entry_s:
        t_to_entry = (pit_entry_s - s_target) / v_target_ms
    else:
        t_to_entry = ((track.length - s_target) / v_target_ms) + (pit_entry_s / avg_v_target)

    t_pit_lane = (pit_entry_s - pit_exit_s) / avg_v_target + pit_loss_s
    t_target_now_est = float(np.interp(s_target, track.s, target_car["t_car"]))
    target_exit_abs_t = t_target_now_est + t_to_entry + t_pit_lane

    ahead_count = 0
    nearest_diff = None
    for idx, other in enumerate(cars):
        if idx == car_idx:
            continue
        t_other_exit = float(np.interp(pit_exit_s, track.s, other["t_car"]))
        s_other_now = float(other["ss"][frame_idx])
        if s_other_now > pit_exit_s:
            t_other_exit += other["lap_time"]
        if t_other_exit < target_exit_abs_t:
            ahead_count += 1
        diff = target_exit_abs_t - t_other_exit
        if nearest_diff is None or abs(diff) < abs(nearest_diff):
            nearest_diff = diff

    return ahead_count + 1, nearest_diff


def main():
    vehicle, tire = Vehicle(), Tire()
    track = load_circuit()
    base_result = simulate_flying_lap(vehicle, tire, track)

    x, y = track_xy(track)
    base_v_kmh = base_result.speed * 3.6
    times, cars = build_car_profiles(base_result, track, x, y)
    max_t = float(times[-1])

    pit_entry_s = track.length * PIT_ENTRY_FRAC
    pit_exit_s = track.length * PIT_EXIT_FRAC

    # --- FENÊTRE 1 : Carte, Profil de vitesse, Gap Map & Contrôles ---
    fig = plt.figure(figsize=(18.5, 8.8))
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[4.3, 1.35, 0.55],
        width_ratios=[1.25, 1.35],
        hspace=0.42,
        wspace=0.22,
    )

    ax_map = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[1, :])

    ax_slider = fig.add_axes([0.11, 0.03, 0.36, 0.03])
    ax_play = fig.add_axes([0.49, 0.025, 0.07, 0.04])
    ax_back = fig.add_axes([0.57, 0.025, 0.06, 0.04])
    ax_fwd = fig.add_axes([0.64, 0.025, 0.06, 0.04])
    ax_target = fig.add_axes([0.72, 0.025, 0.09, 0.04])
    ax_pit_minus = fig.add_axes([0.82, 0.025, 0.065, 0.04])
    ax_pit_plus = fig.add_axes([0.895, 0.025, 0.065, 0.04])

    # --- FENÊTRE 2 : Scoreboards dédiés ---
    fig_sb, (ax_sb_main, ax_sb_sec) = plt.subplots(
        1, 2, figsize=(15.5, 7.6), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig_sb.canvas.manager.set_window_title("Scoreboards - F1 2025")
    fig_sb.patch.set_facecolor("#0E1117")

    ax_sb_main.set_facecolor("#0E1117")
    ax_sb_main.axis("off")
    ax_sb_main.set_title("LIVE SCOREBOARD & PIT EXIT EST.", color="white", fontsize=12, pad=10)
    sb_main_text = ax_sb_main.text(
        0.01, 0.97, "", transform=ax_sb_main.transAxes,
        va="top", ha="left", family="monospace", fontsize=9.2, color="#E6EDF3",
    )

    ax_sb_sec.set_facecolor("#0E1117")
    ax_sb_sec.axis("off")
    ax_sb_sec.set_title("SECTORS & LAP SUMMARY (S1 / S2 / S3 / LAP)", color="white", fontsize=12, pad=10)
    sb_sec_text = ax_sb_sec.text(
        0.01, 0.97, "", transform=ax_sb_sec.transAxes,
        va="top", ha="left", family="monospace", fontsize=9.2, color="#E6EDF3",
    )

    # 1. Carte du circuit
    sc = ax_map.scatter(x, y, c=base_v_kmh, s=8, cmap="RdYlGn", alpha=0.75)
    fig.colorbar(sc, ax=ax_map, label="Vitesse de base [km/h]")
    ax_map.set_aspect("equal")
    ax_map.set_title(f"{CIRCUIT.upper()} - Saison 2025 (Victoire {cars[0]['code']})")

    map_trails = []
    map_markers = []
    map_labels = []
    for car in cars:
        is_lando = car["code"] == "NOR"
        trail, = ax_map.plot(
            [], [], "-",
            linewidth=2.4 if is_lando else 1.2,
            alpha=0.75 if is_lando else 0.4,
            color=car["color"],
            zorder=4 if is_lando else 2,
        )
        marker, = ax_map.plot(
            [], [], "o",
            ms=11 if is_lando else 7,
            color=car["color"],
            markeredgecolor="black",
            markeredgewidth=1.4 if is_lando else 0.7,
            zorder=5 if is_lando else 3,
        )
        label = ax_map.text(
            0, 0, car["code"],
            fontsize=8 if is_lando else 7,
            fontweight="bold" if is_lando else "normal",
            color="black",
            zorder=6 if is_lando else 4,
        )
        map_trails.append(trail)
        map_markers.append(marker)
        map_labels.append(label)

    time_box = ax_map.text(
        0.02, 0.97, "", transform=ax_map.transAxes,
        va="top", fontsize=9.8,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92),
    )

    # 2. Profil de vitesse + repères entrée/sortie des stands
    ax_v.plot(track.s, base_v_kmh, color="gray", alpha=0.45, linewidth=1.2)
    ax_v.axvline(pit_entry_s, color="#E67E22", linestyle="--", linewidth=1.2, label="Pit Entry")
    ax_v.axvline(pit_exit_s, color="#27AE60", linestyle="--", linewidth=1.2, label="Pit Exit")
    ax_v.set_xlabel("Distance [m]")
    ax_v.set_ylabel("Vitesse [km/h]")
    ax_v.set_title("Profil de vitesse & Pit Lane")
    ax_v.grid(True)
    ax_v.legend(loc="lower right", fontsize=8)

    v_markers = []
    for car in cars:
        is_lando = car["code"] == "NOR"
        pt, = ax_v.plot(
            [], [], "o",
            ms=9 if is_lando else 6,
            color=car["color"],
            markeredgecolor="black",
            markeredgewidth=1.2 if is_lando else 0.6,
            zorder=5 if is_lando else 3,
        )
        v_markers.append(pt)

    # 3. Gap Map
    ax_gap.set_facecolor("#111111")
    ax_gap.axvline(0.0, color="#00FF88", linestyle="--", linewidth=1.4)
    ax_gap.set_xlim(-GAP_WINDOW_S, 1.0)
    ax_gap.set_ylim(-1.2, 1.4)
    ax_gap.set_yticks([])
    ax_gap.set_xticks(np.arange(-GAP_WINDOW_S, 1.5, 1.0))
    ax_gap.tick_params(axis="x", colors="white", labelsize=9)
    ax_gap.set_xlabel("Écart par rapport au leader [s]", color="white")
    ax_gap.set_title("Gap Map - Écarts en temps réel", color="white", fontsize=10.5)
    ax_gap.grid(axis="x", color="#333333", linestyle=":", linewidth=0.8)

    gap_markers = []
    gap_labels = []
    for car in cars:
        is_lando = car["code"] == "NOR"
        gm, = ax_gap.plot(
            [], [], "o",
            ms=11 if is_lando else 8,
            color=car["color"],
            markeredgecolor="white" if not is_lando else "#00FF88",
            markeredgewidth=1.6 if is_lando else 0.8,
            zorder=6 if is_lando else 4,
        )
        gl = ax_gap.text(
            0, 0, car["code"],
            color="white",
            fontsize=7.8 if is_lando else 7,
            fontweight="bold" if is_lando else "normal",
            ha="center",
            va="bottom",
            zorder=7 if is_lando else 5,
        )
        gap_markers.append(gm)
        gap_labels.append(gl)

    # Widgets de contrôle
    slider = Slider(ax_slider, "Temps [s]", 0.0, max_t, valinit=0.0, valstep=0.1)
    btn_play = Button(ax_play, "Pause")
    btn_back = Button(ax_back, "-5s")
    btn_fwd = Button(ax_fwd, "+5s")
    btn_target = Button(ax_target, f"Pit: {PIT_TARGET_CODE}")
    btn_pit_minus = Button(ax_pit_minus, "Pit -1s")
    btn_pit_plus = Button(ax_pit_plus, "Pit +1s")

    car_codes = [c["code"] for c in cars]
    state = {
        "playing": True,
        "updating_slider": False,
        "target_idx": car_codes.index(PIT_TARGET_CODE) if PIT_TARGET_CODE in car_codes else 0,
        "pit_loss": PIT_LOSS_S,
        "frame_idx": 0,
    }

    def render_frame(frame_idx):
        frame_idx = int(np.clip(frame_idx, 0, len(times) - 1))
        state["frame_idx"] = frame_idx
        t_now = float(times[frame_idx])

        live_order = sorted(
            range(len(cars)),
            key=lambda idx: (-cars[idx]["ss"][frame_idx], cars[idx]["lap_time"]),
        )
        leader_idx = live_order[0]
        leader_car = cars[leader_idx]
        leader_s = float(leader_car["ss"][frame_idx])
        leader_v_ms = max(10.0, float(leader_car["vs"][frame_idx]) / 3.6)

        gaps_s = {}
        for idx in range(len(cars)):
            if idx == leader_idx:
                gaps_s[idx] = 0.0
            else:
                car_s = float(cars[idx]["ss"][frame_idx])
                car_v_ms = max(10.0, float(cars[idx]["vs"][frame_idx]) / 3.6)
                avg_v_ms = 0.5 * (leader_v_ms + car_v_ms)
                gaps_s[idx] = max(0.0, leader_s - car_s) / avg_v_ms

        target_idx = state["target_idx"]
        target_car = cars[target_idx]
        pit_loss_now = state["pit_loss"]
        est_pos, nearest_diff = estimate_pit_pos_for_car(
            target_idx, cars, track, frame_idx, pit_loss_now
        )
        diff_str = f"{nearest_diff:+.2f}s" if nearest_diff is not None else "N/A"
        pit_summary = (
            f"PIT EXIT ({target_car['code']}): P{est_pos} est. | "
            f"Perte {pit_loss_now:.0f}s | Ecart: {diff_str}"
        )

        time_box.set_text(
            f"t = {t_now:5.1f} / {max_t:5.1f} s\n"
            f"Leader : {leader_car['code']}\n"
            f"{pit_summary}"
        )

        start_idx = max(0, frame_idx - TRAIL_FRAMES)
        main_lines = [
            "P  PIL  TEAM         GAP      KM/H  PIT_EST",
            "-" * 46,
        ]
        sec_lines = [
            "PIL  S1      S2      S3      LAP_EST   VMAX",
            "-" * 45,
        ]

        for rank, idx in enumerate(live_order, start=1):
            car = cars[idx]
            xi = float(car["xs"][frame_idx])
            yi = float(car["ys"][frame_idx])
            si = float(car["ss"][frame_idx])
            vi = float(car["vs"][frame_idx])
            gap = gaps_s[idx]

            map_trails[idx].set_data(
                car["xs"][start_idx:frame_idx + 1],
                car["ys"][start_idx:frame_idx + 1],
            )
            map_markers[idx].set_data([xi], [yi])
            map_labels[idx].set_position((xi + 12, yi + 12))
            v_markers[idx].set_data([si], [vi])

            x_gap = float(np.clip(-gap, -GAP_WINDOW_S + 0.2, 0.5))
            y_gap = -0.45 + ((rank - 1) % 5) * 0.25
            gap_markers[idx].set_data([x_gap], [y_gap])
            gap_labels[idx].set_position((x_gap, y_gap + 0.08))

            gap_str = "LEADER " if rank == 1 else f"+{gap:5.2f}s"
            car_pit_pos, _ = estimate_pit_pos_for_car(idx, cars, track, frame_idx, pit_loss_now)
            star = "*" if car["code"] == "NOR" else " "

            main_lines.append(
                f"{rank:2d} {car['code']:3s}{star} {car['team'][:11]:11s} "
                f"{gap_str:7s} {vi:4.0f}   P{car_pit_pos:2d}"
            )
            sec_lines.append(
                f"{car['code']:3s}{star} {car['s1']:6.2f}  {car['s2']:6.2f}  "
                f"{car['s3']:6.2f}  {car['lap_time']:7.2f}s  {car['v_max_kmh']:3.0f}"
            )

        main_lines.append("-" * 46)
        main_lines.append(pit_summary)
        sb_main_text.set_text("\n".join(main_lines))
        sb_sec_text.set_text("\n".join(sec_lines))

        state["updating_slider"] = True
        slider.set_val(t_now)
        state["updating_slider"] = False

        fig.canvas.draw_idle()
        fig_sb.canvas.draw_idle()

    def on_slider_change(val):
        if state["updating_slider"]:
            return
        idx = int(np.argmin(np.abs(times - float(val))))
        render_frame(idx)

    def toggle_play(event):
        state["playing"] = not state["playing"]
        btn_play.label.set_text("Pause" if state["playing"] else "Play")
        if state["playing"]:
            anim.event_source.start()
        else:
            anim.event_source.stop()

    def jump_time(delta_s):
        new_t = float(np.clip(slider.val + delta_s, 0.0, max_t))
        idx = int(np.argmin(np.abs(times - new_t)))
        render_frame(idx)

    def cycle_target(_event):
        state["target_idx"] = (state["target_idx"] + 1) % len(cars)
        code = cars[state["target_idx"]]["code"]
        btn_target.label.set_text(f"Pit: {code}")
        render_frame(state["frame_idx"])

    def adjust_pit_loss(delta_s):
        state["pit_loss"] = float(np.clip(state["pit_loss"] + delta_s, 5.0, 45.0))
        render_frame(state["frame_idx"])

    def on_key(event):
        if event.key == "right":
            jump_time(2.0)
        elif event.key == "left":
            jump_time(-2.0)
        elif event.key == " ":
            toggle_play(event)

    slider.on_changed(on_slider_change)
    btn_play.on_clicked(toggle_play)
    btn_back.on_clicked(lambda _e: jump_time(-5.0))
    btn_fwd.on_clicked(lambda _e: jump_time(5.0))
    btn_target.on_clicked(cycle_target)
    btn_pit_minus.on_clicked(lambda _e: adjust_pit_loss(-1.0))
    btn_pit_plus.on_clicked(lambda _e: adjust_pit_loss(1.0))
    fig.canvas.mpl_connect("key_press_event", on_key)

    def update(frame_idx):
        if state["playing"]:
            render_frame(frame_idx)
        return []

    anim = FuncAnimation(
        fig, update, frames=len(times),
        interval=1000 / FPS, repeat=True, blit=False,
    )
    plt.show()
    return anim


if __name__ == "__main__":
    main()