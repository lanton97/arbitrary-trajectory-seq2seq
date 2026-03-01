"""
Read ROS2 MCAP bag files and compare true poses against unicycle model
predictions driven by cmd_vel. No ROS installation required.

Usage:
    python scripts/read_bag.py <bag_dir_or_mcap_file> [options]

Examples:
    # List topics
    python scripts/read_bag.py convoy_bags/final_rosbags/exp_1_a_2025sep09 --list-topics

    # Compare with defaults (auto-detects pose and cmd_vel topics)
    python scripts/read_bag.py convoy_bags/final_rosbags/exp_1_a_2025sep09

    # Specify topics explicitly
    python scripts/read_bag.py convoy_bags/final_rosbags/exp_1_a_2025sep09 \
        --pose-topic /vrpn_mocap/Husky_quarrg/pose \
        --cmd-topic /a300_00013/cmd_vel

    # Batch mode: process all subdirectories and print an error summary
    python scripts/read_bag.py convoy_bags/final_rosbags --batch
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mcap_ros2.reader import read_ros2_messages


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_mcap(path: str) -> Path:
    p = Path(path)
    if p.is_file() and p.suffix == ".mcap":
        return p
    if p.is_dir():
        mcaps = sorted(p.glob("*.mcap"))
        if not mcaps:
            sys.exit(f"No .mcap files found in {p}")
        return mcaps[0]
    sys.exit(f"Path not found or not an .mcap file/directory: {path}")


def list_topics(mcap_path: Path):
    from mcap.reader import make_reader
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
    print(f"\nTopics in {mcap_path.name}:")
    print(f"  {'Topic':<55} {'Type':<45} {'Messages'}")
    print("  " + "-" * 110)
    if summary and summary.channels:
        count_map = {}
        if summary.statistics and summary.statistics.channel_message_counts:
            count_map = dict(summary.statistics.channel_message_counts)
        for ch_id, ch in sorted(summary.channels.items(), key=lambda x: x[1].topic):
            schema = summary.schemas.get(ch.schema_id)
            msgtype = schema.name if schema else "unknown"
            count = count_map.get(ch_id, "?")
            print(f"  {ch.topic:<55} {msgtype:<45} {count}")


def quat_to_yaw(qx, qy, qz, qw) -> float:
    """Extract yaw (rotation about Z) from quaternion."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def read_poses(mcap_path: Path, topic: str):
    """Return list of (timestamp_s, x, y, yaw)."""
    records = []
    for msg in read_ros2_messages(str(mcap_path), topics=[topic]):
        m = msg.ros_msg
        t = msg.log_time_ns * 1e-9
        p = m.pose.position
        o = m.pose.orientation
        yaw = quat_to_yaw(o.x, o.y, o.z, o.w)
        records.append((t, p.x, p.y, yaw))
    records.sort(key=lambda r: r[0])
    return records


def read_cmd_vel(mcap_path: Path, topic: str):
    """Return list of (timestamp_s, v, omega)."""
    records = []
    for msg in read_ros2_messages(str(mcap_path), topics=[topic]):
        m = msg.ros_msg
        t = msg.log_time_ns * 1e-9
        # TwistStamped has .twist.linear.x / .twist.angular.z
        try:
            v = m.twist.linear.x
            w = m.twist.angular.z
        except AttributeError:
            # plain Twist
            v = m.linear.x
            w = m.angular.z
        records.append((t, v, w))
    records.sort(key=lambda r: r[0])
    return records


# ---------------------------------------------------------------------------
# Unicycle integration
# ---------------------------------------------------------------------------

def integrate_unicycle(poses, cmd_vels):
    """
    Integrate a unicycle model over cmd_vel commands, starting from the
    first true pose. Returns arrays of (t, x, y, yaw) for each cmd_vel step.

    The robot model:
        dx/dt    = v * cos(theta)
        dy/dt    = v * sin(theta)
        dtheta/dt = omega

    cmd_vel messages are treated as zero-order hold (constant between steps).
    """
    if not poses or not cmd_vels:
        return []

    # Start from the first pose that overlaps with cmd_vel time window
    t_cmd_start = cmd_vels[0][0]

    # Find the pose closest to the first cmd_vel
    init_pose = min(poses, key=lambda p: abs(p[0] - t_cmd_start))
    _, x0, y0, th0 = init_pose

    x, y, th = x0, y0, th0
    integrated = [(t_cmd_start, x, y, th)]

    for i in range(len(cmd_vels) - 1):
        t0, v, w  = cmd_vels[i]
        t1        = cmd_vels[i + 1][0]
        dt        = t1 - t0
        if dt <= 0:
            continue
        # Euler integration
        x  += v * math.cos(th) * dt
        y  += v * math.sin(th) * dt
        th += w * dt
        th  = math.atan2(math.sin(th), math.cos(th))   # wrap to [-pi, pi]
        integrated.append((t1, x, y, th))

    return integrated


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(true_poses, predicted, pose_topic, cmd_topic, bag_name):
    if not true_poses or not predicted:
        print("Not enough data to plot.")
        return

    # Filter true poses to cmd_vel time window
    t_start = predicted[0][0]
    t_end   = predicted[-1][0]
    true_window = [(t, x, y, th) for t, x, y, th in true_poses
                   if t_start <= t <= t_end]
    if not true_window:
        print("No true poses overlap with cmd_vel time window.")
        true_window = true_poses  # fall back to all

    # Unpack
    t_true = np.array([r[0] for r in true_window]) - t_start
    x_true = np.array([r[1] for r in true_window])
    y_true = np.array([r[2] for r in true_window])
    th_true = np.array([r[3] for r in true_window])

    t_pred = np.array([r[0] for r in predicted]) - t_start
    x_pred = np.array([r[1] for r in predicted])
    y_pred = np.array([r[2] for r in predicted])
    th_pred = np.array([r[3] for r in predicted])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{bag_name}\nPose: {pose_topic} | Cmd: {cmd_topic}", fontsize=10)

    # XY trajectory
    ax = axes[0, 0]
    ax.plot(x_true, y_true, label="True pose", linewidth=2)
    ax.plot(x_pred, y_pred, "--", label="Unicycle prediction", linewidth=2)
    # Mark start
    ax.plot(x_true[0], y_true[0], "go", markersize=8, label="Start")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("XY Trajectory")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True)

    # X vs time
    ax = axes[0, 1]
    ax.plot(t_true, x_true, label="True x")
    ax.plot(t_pred, x_pred, "--", label="Predicted x")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("x [m]")
    ax.set_title("x vs Time")
    ax.legend()
    ax.grid(True)

    # Y vs time
    ax = axes[1, 0]
    ax.plot(t_true, y_true, label="True y")
    ax.plot(t_pred, y_pred, "--", label="Predicted y")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("y [m]")
    ax.set_title("y vs Time")
    ax.legend()
    ax.grid(True)

    # Yaw vs time
    ax = axes[1, 1]
    ax.plot(t_true, np.degrees(th_true), label="True yaw")
    ax.plot(t_pred, np.degrees(th_pred), "--", label="Predicted yaw")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Yaw [deg]")
    ax.set_title("Yaw vs Time")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    out_path = Path(f"bag_comparison_{bag_name}.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")
    plt.show()




# ---------------------------------------------------------------------------
# Topic auto-detection
# ---------------------------------------------------------------------------

POSE_KEYWORDS  = ["pose"]
CMD_VEL_KEYWORDS = ["cmd_vel"]

def autodetect_topic(topics, keywords, label: str) -> str:
    candidates = [t for t in topics if any(k in t for k in keywords)]
    if not candidates:
        sys.exit(f"Could not auto-detect {label} topic. Use --list-topics then pass it explicitly.")
    if len(candidates) == 1:
        return candidates[0]
    # Prefer topics that contain 'pose' or 'cmd_vel' (not 'twist')
    print(f"\nMultiple {label} topics found:")
    for i, t in enumerate(candidates):
        print(f"  [{i}] {t}")
    choice = input(f"Select {label} topic index [0]: ").strip()
    idx = int(choice) if choice else 0
    return candidates[idx]


def get_all_topics(mcap_path: Path):
    from mcap.reader import make_reader
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
    if summary and summary.channels:
        return [ch.topic for ch in summary.channels.values()]
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_errors(poses, predicted):
    """Return (pos_error_m, heading_error_deg) for the final timestep."""
    if not poses or not predicted:
        return None, None
    true_final = np.array(poses[-1][1:4])   # (x, y, th)
    pred_final = np.array(predicted[-1][1:4])
    pos_error = float(np.linalg.norm(true_final[:2] - pred_final[:2]))
    th_error  = float(np.degrees(np.arctan2(
        np.sin(true_final[2] - pred_final[2]),
        np.cos(true_final[2] - pred_final[2])
    )))
    return pos_error, th_error


def process_bag(bag_path: str, pose_topic=None, cmd_topic=None, show_plot=True):
    """
    Process a single bag directory / mcap file.
    Returns (pos_error, heading_error) or (None, None) on failure.
    """
    try:
        mcap_path = find_mcap(bag_path)
    except SystemExit as e:
        print(f"  Skipping: {e}")
        return None, None

    print(f"Reading: {mcap_path}")
    all_topics = get_all_topics(mcap_path)

    try:
        pt = pose_topic or autodetect_topic(all_topics, POSE_KEYWORDS, "pose")
        ct = cmd_topic  or autodetect_topic(all_topics, CMD_VEL_KEYWORDS, "cmd_vel")
    except SystemExit as e:
        print(f"  Topic error: {e}")
        return None, None

    print(f"  Pose topic : {pt}")
    print(f"  Cmd topic  : {ct}")

    poses    = read_poses(mcap_path, pt)
    cmd_vels = read_cmd_vel(mcap_path, ct)
    predicted = integrate_unicycle(poses, cmd_vels)

    print(f"  {len(poses)} poses, {len(cmd_vels)} cmd_vels, {len(predicted)} integration steps")

    bag_name = mcap_path.parent.name if mcap_path.parent.name != "." else mcap_path.stem
    if show_plot:
        plot_comparison(poses, predicted, pt, ct, bag_name)

    pos_err, th_err = compute_errors(poses, predicted)
    if pos_err is not None:
        print(f"  Final position error : {pos_err:.3f} m")
        print(f"  Final heading error  : {th_err:.2f} deg")
    return pos_err, th_err


def main():
    parser = argparse.ArgumentParser(description="Read ROS2 MCAP bags and compare poses vs unicycle prediction")
    parser.add_argument("bag", help="Path to bag directory or .mcap file (or parent dir in --batch mode)")
    parser.add_argument("--list-topics", action="store_true", help="List topics and exit")
    parser.add_argument("--pose-topic", default=None, help="Pose topic (PoseStamped)")
    parser.add_argument("--cmd-topic",  default=None, help="Command velocity topic (TwistStamped)")
    parser.add_argument("--batch", action="store_true",
                        help="Iterate over all subdirectories of <bag>, report a final error summary table")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting (useful in batch mode)")
    args = parser.parse_args()

    if args.batch:
        parent = Path(args.bag)
        if not parent.is_dir():
            sys.exit(f"--batch requires a directory, got: {parent}")

        subdirs = sorted(d for d in parent.iterdir() if d.is_dir())
        if not subdirs:
            sys.exit(f"No subdirectories found in {parent}")

        results = []  # list of (name, pos_err, th_err)
        for subdir in subdirs:
            print(f"\n{'='*60}")
            print(f"Bag: {subdir.name}")
            pos_err, th_err = process_bag(
                str(subdir),
                pose_topic=args.pose_topic,
                cmd_topic=args.cmd_topic,
                show_plot=not args.no_plot,
            )
            results.append((subdir.name, pos_err, th_err))

        # Summary table
        print(f"\n{'='*60}")
        print("BATCH SUMMARY")
        print(f"{'='*60}")
        col_w = max(len(r[0]) for r in results) + 2
        header = f"{'Bag':<{col_w}}  {'Pos error (m)':>14}  {'Heading error (deg)':>20}"
        print(header)
        print("-" * len(header))
        for name, pos_err, th_err in results:
            pe = f"{pos_err:.3f}" if pos_err is not None else "N/A"
            te = f"{th_err:.2f}"  if th_err  is not None else "N/A"
            print(f"{name:<{col_w}}  {pe:>14}  {te:>20}")
        print(f"{'Mean position error:':<{col_w}}  {np.mean([r[1] for r in results if r[1] is not None]):.3f} m")
        print(f"{'Mean heading error:':<{col_w}}  {np.mean([r[2] for r in results if r[2] is not None]):.2f} deg")

        return

    # Single-bag mode
    mcap_path = find_mcap(args.bag)
    print(f"Reading: {mcap_path}")

    if args.list_topics:
        list_topics(mcap_path)
        return

    all_topics = get_all_topics(mcap_path)
    pose_topic = args.pose_topic or autodetect_topic(all_topics, POSE_KEYWORDS, "pose")
    cmd_topic  = args.cmd_topic  or autodetect_topic(all_topics, CMD_VEL_KEYWORDS, "cmd_vel")

    print(f"Pose topic : {pose_topic}")
    print(f"Cmd topic  : {cmd_topic}")

    print("Reading pose data...")
    poses = read_poses(mcap_path, pose_topic)
    print(f"  {len(poses)} pose messages")

    print("Reading cmd_vel data...")
    cmd_vels = read_cmd_vel(mcap_path, cmd_topic)
    print(f"  {len(cmd_vels)} cmd_vel messages")

    print("Integrating unicycle model...")
    predicted = integrate_unicycle(poses, cmd_vels)
    print(f"  {len(predicted)} integration steps")

    bag_name = mcap_path.parent.name if mcap_path.parent.name != "." else mcap_path.stem
    if not args.no_plot:
        plot_comparison(poses, predicted, pose_topic, cmd_topic, bag_name)

    pos_err, th_err = compute_errors(poses, predicted)
    if pos_err is not None:
        print(f"\nFinal position error: {pos_err:.3f} m")
        print(f"Final heading error: {th_err:.2f} deg")


if __name__ == "__main__":
    main()
