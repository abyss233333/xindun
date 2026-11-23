"""Shared follow-up planning for real_grasp and its camera animation.

Station poses and step numbers follow xindun_1110-2/main_new(1)cc.py.
All IK is recomputed with the current I5Dual model and current bottle grasp.
No reference JSON is modified or replayed as robot joint commands.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Barrier

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import basis.robot_math as rm
import modeling.collision_model as cm
from modeling.solid_mesh_collision import containment_contacts
import motion.probabilistic.rrt_connect as rrtc
from motion.trajectory.piecewisepoly_toppra import PiecewisePolyTOPPRA

RIGHT_START = np.array([-0.70687932, -0.28622089, -2.3381843,
                        -0.90192553, -2.56188762, -6.1588563 + 2 * math.pi])
LEFT_HOME = np.array([
    0.44462849, 0.02638729, 1.59294087,
    1.33750434, 2.58590077, -1.28399963,
])
RIGHT_HOME = np.array([
    -0.76378101, -0.55498721, -1.79941327,
    0.01021574, -2.0996724, 0.40663958,
])
SAMPLE_PERIOD = 0.005
LINEAR_SPACING = 0.001
LINEAR_ANGULAR_SPACING = 0.01  # Radians per IK point when returning with a rotation.
RIGHT_CLEARANCE_Y = 0.10  # Metres along world -Y after opening the left gripper.
OPENER_PUSH_X = 0.100  # Metres along world +X after reaching the original approach pose.
HOLD_WIDTH = 0.09  # Simulation only: the modified real gripper uses close_g().
TOOL_WIDTH = 0.03
OBJECT_DIR = os.path.join(os.path.dirname(__file__), "objects")
TABLE_CLEARANCE_BELOW_OBJECT = 0.003
TABLE_OFFSET_WORLD_XY = np.array([0.50, -0.40])
TABLE_IGNORED_ROBOT_LABEL_PREFIXES = (
    "lft_arm/link", "rgt_arm/link", "lft_hnd/", "rgt_hnd/",
)
OPENER_COLLISION_POSITIONS = ([1.0, .200, 0.], [1.0, -.050, 0.],
                              [1.0, -.300, 0.])


def homo(pos, rot):
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = rot, pos
    return result


def tcp_pose(robot, arm):
    return homo(*robot.get_gl_tcp(arm))


def rotation_error(a, b):
    return float(np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)))


def right_clearance_target(start_tcp):
    """Shift in world -Y; point TCP +Z down, preserving its horizontal heading."""
    x_axis = start_tcp[:3, 0].copy()
    x_axis[2] = 0.
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1., 0., 0.])
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0., 0., -1.])
    rotation = np.column_stack((x_axis, np.cross(z_axis, x_axis), z_axis))
    return homo(start_tcp[:3, 3] + [0., -RIGHT_CLEARANCE_Y, 0.], rotation)


def make_opener_obstacles():
    """Create the three current opener meshes used as planning obstacles."""
    obstacles = []
    opener_rot = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    for position in OPENER_COLLISION_POSITIONS:
        opener = cm.CollisionModel(os.path.join(OBJECT_DIR, "bottle_openning.stl"))
        opener.set_homomat(homo(position, opener_rot))
        obstacles.append(opener)
    return obstacles


def make_table_obstacle(object_pose, target):
    """Place the table top 2 mm below the transformed object's lowest point."""
    table = cm.CollisionModel(os.path.join(OBJECT_DIR, "table.stl"))
    table._xindun_table_obstacle = True
    local_vertices = np.asarray(target.objtrm.vertices, dtype=float)
    world_vertices = (local_vertices @ object_pose[:3, :3].T
                      + object_pose[:3, 3])
    object_bottom_z = float(np.min(world_vertices[:, 2]))
    table_bounds = np.asarray(table.objtrm.bounds, dtype=float)
    table_rot = rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)
    table_center_xy = table_rot[:2, :2] @ np.mean(table_bounds[:, :2], axis=0)
    table_top_local_z = float(table_bounds[1, 2])
    table_center_world_xy = object_pose[:2, 3] + TABLE_OFFSET_WORLD_XY
    table_pos = np.array([
        table_center_world_xy[0] - table_center_xy[0],
        table_center_world_xy[1] - table_center_xy[1],
        object_bottom_z - TABLE_CLEARANCE_BELOW_OBJECT - table_top_local_z,
    ])
    table.set_pos(table_pos)
    table.set_rotmat(table_rot)
    table.set_rgba([.55, .55, .58, 1.])
    return table


def plan_right_clearance(robot, target, obstacles=None):
    """Plan while the left arm is at hover with its gripper open."""
    start = robot.get_jnt_values("rgt_arm").copy()
    action = dict(arm="rgt_arm", kind="joint", event=None,
                  label="右臂向世界 -Y 让位，夹爪朝下",
                  target=right_clearance_target(tcp_pose(robot, "rgt_arm")),
                  obstacles=[] if obstacles is None else list(obstacles))
    print(f"规划右臂让位：世界 Y 减少 {RIGHT_CLEARANCE_Y * 1000:g} mm，TCP +Z 朝世界 -Z。",
          flush=True)
    try:
        action["path"] = solve_path(robot, target, action)
        actual = tcp_pose(robot, "rgt_arm")
        if (np.linalg.norm(actual[:3, 3] - action["target"][:3, 3]) > .001
                or rotation_error(actual[:3, :3], action["target"][:3, :3]) > .002):
            raise RuntimeError("右臂让位终点的 TCP 位姿误差超限。")
        return action
    except Exception:
        robot.fk("rgt_arm", start)
        robot.get_oih_list()
        raise


def station_actions(plan, check_opener_collisions=True):
    """Return TCP targets, keeping stations fixed and placement camera-relative."""
    # cf.stl's neck points along object +X (the old flask used +Y).
    # At the opener: bottle +X -> world +Z, bottle +Z -> world -X.
    # With the current Rx(pi) grasp, the gripper face (TCP +Z) points +X.
    opener_rot = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 2)
    grasp = plan["jaw_to_object"]
    opener_approach = homo([.8, .2, .20], opener_rot) @ grasp
    opener_low = opener_approach.copy()
    opener_low[0, 3] += OPENER_PUSH_X
    opener_high = opener_low.copy()
    opener_high[2, 3] += .035
    # During right-arm operation: bottle +X -> world -Y, bottle +Z stays up.
    work_rot = rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2)
    bottle_work = homo([.6, .0, .2], work_rot) @ grasp
    # Right TCP poses are the explicit world poses in the reference source.
    tilt = rm.rotmat_from_axangle([0, 1, 0], -math.pi / 9)
    insertion_rot = np.array([[-1., 0., 0.],
                              [0., math.sin(math.pi / 9), math.cos(math.pi / 9)],
                              [0., math.cos(math.pi / 9), -math.sin(math.pi / 9)]]) @ tilt
    vertical_rot = np.diag([-1., 1., -1.]) @ tilt
    pipette_near = homo([.5, -.17603508 - .05 * math.cos(math.pi / 9),
                         .25247232 + .05 * math.sin(math.pi / 9)], insertion_rot)
    pipette_in = homo([.5, -.17603508, .25247232], insertion_rot)
    site_a_high = homo([.51199956, -.27600033, .283], vertical_rot)
    site_a_low = homo([.51199956, -.27600033, .183], vertical_rot)
    site_b_high = homo([.51199956, -.37600033, .283], vertical_rot)
    site_b_low = homo([.51199956, -.37600033, .183], vertical_rot)
    pre = homo(plan["pregrasp_pos"], (plan["object_pose"] @ grasp)[:3, :3])
    down = plan["object_pose"] @ grasp
    # Follow plan_linear_actions.py's motion types. Step 22 returns to a live
    # orientation, so its straight TCP path must also interpolate rotation.
    table_obstacle = plan["table_obstacle"]
    opener_obstacles = make_opener_obstacles() if check_opener_collisions else []
    specs = [
        (5, "lft_arm", "path", opener_low, "持瓶沿 RRTC 路径到开瓶位置", None, "rrtc"),
        (6, "lft_arm", "linear", opener_high, "上移 40 mm 到开瓶位置", None),
        (7, "lft_arm", "linear", opener_low, "退出开瓶位置", None),
        (7, "lft_arm", "linear", opener_approach, f"持瓶沿世界 -X 退回 {OPENER_PUSH_X * 1000:g} mm", None),
        (8, "lft_arm", "path", bottle_work, "持瓶到移液位置", None),
        (9, "rgt_arm", "path", pipette_near, "移液枪接近培养瓶", None),
        (10, "rgt_arm", "linear", pipette_in, "移液枪进入培养瓶", None),
        (11, "rgt_arm", "linear", pipette_near, "移液枪退出培养瓶", None),
        (12, "rgt_arm", "path", site_a_high, "移液枪到参考位置 A 上方", None),
        (13, "rgt_arm", "linear", site_a_low, "移液枪下探位置 A", None),
        (14, "rgt_arm", "linear", site_a_high, "移液枪抬离位置 A", None),
        (15, "rgt_arm", "linear", site_b_high, "移液枪平移到位置 B", None),
        (16, "rgt_arm", "linear", site_b_low, "移液枪下探位置 B", None),
        (17, "rgt_arm", "linear", site_b_high, "移液枪抬离位置 B", None),
        (18, "rgt_arm", "path", pipette_near, "移液枪返回培养瓶附近", None),
        (19, "rgt_arm", "linear", pipette_in, "移液枪再次进入培养瓶", None),
        (20, "rgt_arm", "linear", pipette_near, "移液枪再次退出培养瓶", None),
        (21, "rgt_arm", "path", site_a_high, "移液枪退到位置 A 上方", None),
        (22, "lft_arm", "linear", pre, "持瓶直线返回本次识别位置上方", None),
        (23, "lft_arm", "linear", down, "放回瓶子并松开", "release"),
        (24, "lft_arm", "linear", pre, "空爪抬到预抓取点", None),
        (25, "lft_arm", "linear", down, "再次下探并夹住瓶子", "hold"),
        (26, "lft_arm", "linear", pre, "再次持瓶抬升", None),
        (27, "lft_arm", "path", opener_low, "再次沿 RRTC 路径到开瓶位置", None, "rrtc"),
        (28, "lft_arm", "linear", opener_high, "再次上移到开瓶位置", None),
        (29, "lft_arm", "linear", opener_low, "退出并保持夹持", None),
    ]
    actions = []
    for spec in specs:
        if len(spec) == 7:
            s, a, k, t, l, e, planner = spec
        else:
            s, a, k, t, l, e = spec
            planner = None
        actions.append(dict(step=s, arm=a, kind=k, target=t, label=l, event=e,
                            planner=planner,
                            obstacles=[table_obstacle] + (opener_obstacles
                                                          if planner == "rrtc" else []),
                            ignore_obstacle_pairs=(
                                [("lft_arm/held_object", 0)]
                                if s in (23, 24, 25, 26) else []),
                            interpolate_rotation=(s in (6, 22, 28))))
    # Keep step 7's exit to the low opener pose, but remove its separate
    # world -X retreat. Step 8 then continues directly with RRTC.
    actions = [action for action in actions
               if not (action["step"] == 7
                       and np.allclose(action["target"], opener_approach))]
    for action in actions:
        if action["step"] == 8:
            action["planner"] = "rrtc"
            action["obstacles"] = [table_obstacle] + opener_obstacles
    return actions


def make_pipette(robot):
    tool = cm.CollisionModel(os.path.join(OBJECT_DIR, "pipette_gun.stl"))
    # Preserve the reference tool-to-jaw mounting, not just the mesh's long axis.
    # At RIGHT_START its old jaw rotation was Rz(pi/4) @ Rx(pi), while the
    # mesh rotation in main_new(1)cc.py was Rx(-pi/2) @ Rz(pi/9).
    old_jaw_rot = (rm.rotmat_from_axangle([0, 0, 1], math.pi / 4)
                   @ rm.rotmat_from_axangle([1, 0, 0], math.pi))
    old_tool_rot = (rm.rotmat_from_axangle([1, 0, 0], -math.pi / 2)
                    @ rm.rotmat_from_axangle([0, 0, 1], math.pi / 9))
    pose = tcp_pose(robot, "rgt_arm") @ homo([0., 0., 0.], old_jaw_rot.T @ old_tool_rot)
    tool.set_homomat(pose)
    tool.set_rgba([1., .8, .1, 1.])
    robot.hold("rgt_arm", tool, TOOL_WIDTH)
    return tool


def apply_event(robot, target, event):
    if event == "release":
        robot.get_oih_list()
        robot.release("lft_arm", target, .10)
    elif event == "hold":
        robot.hold("lft_arm", target, HOLD_WIDTH)
    robot.get_oih_list()


def collides_with_obstacles(robot, obstacles, otherrobot_list=None,
                            ignore_obstacle_pairs=()):
    """Use exact meshes for CollisionModel obstacles after the box broad phase."""
    obstacles = [] if obstacles is None else list(obstacles)
    otherrobot_kwargs = ({} if otherrobot_list is None
                         else {"otherrobot_list": otherrobot_list})
    if not obstacles or not all(isinstance(obstacle, cm.CollisionModel)
                                for obstacle in obstacles):
        return robot.is_collided(obstacle_list=obstacles, **otherrobot_kwargs)

    robot.get_oih_list()
    if robot.is_collided(obstacle_list=[], **otherrobot_kwargs):
        return True

    ignore_obstacle_pairs = set(ignore_obstacle_pairs)
    for element in robot.cc.all_cdelements:
        if element.get("collision_model") is None:
            continue
        model = robot._collision_mesh(element)
        label = robot._collision_label(element)
        for obstacle_index, obstacle in enumerate(obstacles):
            if (label, obstacle_index) in ignore_obstacle_pairs:
                continue
            if (getattr(obstacle, "_xindun_table_obstacle", False)
                    and label.startswith(TABLE_IGNORED_ROBOT_LABEL_PREFIXES)):
                continue
            if not model.is_pcdwith(obstacle):
                continue
            if (model.is_mcdwith([obstacle])
                    or containment_contacts(model, obstacle)):
                robot.last_collision_pairs = [(label, f"external_obstacle/{obstacle_index}")]
                return True
    robot.last_collision_pairs = []
    return False


class _MeshObstacleCollisionView:
    """Expose exact external-mesh checks to RRT without changing robot APIs."""
    def __init__(self, robot):
        self.robot = robot

    def __getattr__(self, name):
        return getattr(self.robot, name)

    def is_collided(self, obstacle_list=None, otherrobot_list=None):
        return collides_with_obstacles(self.robot, obstacle_list, otherrobot_list)


def check_configuration(robot, target, arm, q, label, obstacles=None,
                        ignore_obstacle_pairs=()):
    if robot.fk(arm, q) != "succ":
        raise RuntimeError(label + "：关节超出范围。")
    robot.get_oih_list()
    if collides_with_obstacles(robot, obstacles,
                               ignore_obstacle_pairs=ignore_obstacle_pairs):
        raise RuntimeError(f"{label}：机器人/持有物碰撞 {robot.last_collision_pairs}")
    held = any(info["collision_model"] is target for info in robot.lft_oih_infos)
    if not held and robot.is_object_collided(target, contact_hand="lft_arm"):
        raise RuntimeError(f"{label}：目标碰撞 {robot.last_collision_pairs}")


def solve_linear_point(robot, arm, position, rotation, previous):
    """Continue the local IK branch, with bounded TRAC-IK fallback.

    The reference IncrementalNIK uses robot.ik, an iteration-limited solver.
    Using it first avoids depending on a fresh 5 ms TRAC-IK search at every
    millimetre. Never accept a local minimum, branch jump, or inaccurate TCP.
    """
    failures = []
    try:
        for method in ("local", "tracik", "tracik", "tracik"):
            robot.fk(arm, previous)
            kwargs = dict(component_name=arm, tgt_pos=position,
                          tgt_rotmat=rotation, seed_jnt_values=previous.copy())
            if method == "local":
                candidate = robot.ik(**kwargs, max_niter=200, local_minima="end")
            else:
                candidate = robot.tracik(**kwargs, solver_type="Distance")
            if candidate is None:
                failures.append(method + " 未收敛")
                continue
            candidate = np.asarray(candidate, dtype=float)
            if candidate.shape != (6,) or not np.isfinite(candidate).all():
                failures.append(method + " 返回无效关节角")
                continue
            candidate = candidate + 2 * math.pi * np.round((previous - candidate) / (2 * math.pi))
            jump = float(np.max(np.abs(candidate - previous)))
            if not robot.is_jnt_values_in_ranges(arm, candidate):
                failures.append(method + " 关节超限")
                continue
            if jump > .10:
                failures.append(f"{method} 最大关节跳变 {jump:.4f} rad")
                continue
            if robot.fk(arm, candidate) != "succ":
                failures.append(method + " FK 失败")
                continue
            actual = tcp_pose(robot, arm)
            if (np.linalg.norm(actual[:3, 3] - position) > .001
                    or rotation_error(actual[:3, :3], rotation) > .002):
                failures.append(method + " TCP 误差超限")
                continue
            return candidate.copy(), ""
    finally:
        # Both solvers and validation may touch simulation state. Failed
        # candidates must not move either the displayed hand or its object.
        robot.fk(arm, previous)
        robot.get_oih_list()
    return None, "；".join(failures)


def solve_path(robot, target, action):
    arm, label = action["arm"], action["label"]
    start = robot.get_jnt_values(arm).copy()
    start_tcp, goal = tcp_pose(robot, arm), action["target"]
    obstacles = action.get("obstacles", [])
    ignore_obstacle_pairs = action.get("ignore_obstacle_pairs", ())
    if action.get("planner") == "rrtc":
        try:
            check_configuration(robot, target, arm, start, label, obstacles,
                                ignore_obstacle_pairs)
        except RuntimeError as exc:
            raise RuntimeError(f"{label}：RRTC 起点碰撞检查失败：{exc}") from exc
        rng = np.random.default_rng(20261116)
        seeds = [start]
        seeds.extend(start + rng.uniform(-math.pi, math.pi, 6) for _ in range(64))
        goal_conf = None
        for seed in seeds:
            robot.fk(arm, start)
            candidate = robot.tracik(
                tgt_pos=goal[:3, 3], tgt_rotmat=goal[:3, :3],
                component_name=arm, solver_type="Distance", seed_jnt_values=seed)
            if candidate is None:
                continue
            candidate = candidate + 2 * math.pi * np.round((start - candidate) / (2 * math.pi))
            if not robot.is_jnt_values_in_ranges(arm, candidate):
                continue
            try:
                check_configuration(robot, target, arm, candidate, label, obstacles,
                                    ignore_obstacle_pairs)
            except RuntimeError:
                continue
            goal_conf = candidate.copy()
            break
        robot.fk(arm, start)
        robot.get_oih_list()
        if goal_conf is None:
            raise RuntimeError(label + "：开瓶器避障目标没有无碰撞逆解。")
        planner = rrtc.RRTConnect(_MeshObstacleCollisionView(robot))
        path = planner.plan(
            component_name=arm,
            start_conf=start,
            goal_conf=goal_conf,
            obstacle_list=obstacles,
            ext_dist=.1,
            max_iter=12000,
            max_time=60.0,
            smoothing_iterations=80,
        )
        if path is None:
            raise RuntimeError(label + "：RRTC 规划失败。")
        for first, second in zip(path[:-1], path[1:]):
            count = max(1, int(np.ceil(np.max(np.abs(second - first)) / .02)))
            for sample in np.linspace(first, second, count + 1):
                check_configuration(robot, target, arm, sample, label, obstacles,
                                    ignore_obstacle_pairs)
        return path
    if action["kind"] == "linear":
        angle = rotation_error(start_tcp[:3, :3], goal[:3, :3])
        turn = action.get("interpolate_rotation", False)
        if angle > .002 and not turn:
            raise RuntimeError(label + "：直线段的起终点姿态不一致。")
        count = max(1, int(np.ceil(np.linalg.norm(goal[:3, 3] - start_tcp[:3, 3]) / LINEAR_SPACING)))
        if turn:
            count = max(count, int(np.ceil(angle / LINEAR_ANGULAR_SPACING)))
            rotations = rm.rotmat_slerp(start_tcp[:3, :3], goal[:3, :3], count + 1)
        path, previous = [start], start
        for i in range(1, count + 1):
            position = start_tcp[:3, 3] + (goal[:3, 3] - start_tcp[:3, 3]) * i / count
            rotation = rotations[i] if turn else goal[:3, :3]
            q, reason = solve_linear_point(robot, arm, position, rotation, previous)
            if q is None:
                raise RuntimeError(f"{label}：第 {i}/{count} 个直线点无连续逆解。{reason}")
            check_configuration(robot, target, arm, q, label, obstacles,
                                ignore_obstacle_pairs)
            actual = tcp_pose(robot, arm)
            if (np.linalg.norm(actual[:3, 3] - position) > .001
                    or rotation_error(actual[:3, :3], rotation) > .002):
                raise RuntimeError(label + "：直线 TCP 误差超限。")
            path.append(q.copy())
            previous = q
        return np.asarray(path)
    seeds = [start]
    if arm == "rgt_arm":
        seeds.extend([RIGHT_START,
                      np.array([-1.35739975, -.41230409, -1.95445237,
                                -.27639284, -1.06731817, -.56475339]),
                      np.array([-1.0623905, -.35759657, -2.56458994,
                                -1.83039921, -2.19300138, -.60797279])])
    rng = np.random.default_rng(20261114)
    seeds.extend(start + rng.uniform(-math.pi, math.pi, 6) for _ in range(16))
    last_error = label + "：目标无逆解。"
    for seed in seeds:
        robot.fk(arm, start)
        q = robot.tracik(tgt_pos=goal[:3, 3], tgt_rotmat=goal[:3, :3],
                         component_name=arm, solver_type="Distance", seed_jnt_values=seed)
        if q is None:
            continue
        # Select the closest equivalent turn within the real model's limits.
        q = q + 2 * math.pi * np.round((start - q) / (2 * math.pi))
        if not robot.is_jnt_values_in_ranges(arm, q):
            continue
        count = max(2, int(np.ceil(np.max(np.abs(q - start)) / .02)) + 1)
        path = np.linspace(start, q, count)
        try:
            for sample in path:
                check_configuration(robot, target, arm, sample, label, obstacles,
                                    ignore_obstacle_pairs)
        except RuntimeError as exc:
            last_error = str(exc)
            continue
        return path
    raise RuntimeError(last_error + "；未找到可直接 MoveJ 的连续配置。")


def timed_samples(path, vel, acc):
    if np.max(np.abs(path - path[0])) < 1e-9:
        return path[:1].copy()
    return np.asarray(PiecewisePolyTOPPRA().interpolate_by_max_spdacc(
        path=path, control_frequency=SAMPLE_PERIOD, max_vels=[vel] * 6,
        max_accs=[acc] * 6, toggle_debug=False))


def planned_samples(plan, key, path, vel, acc):
    """Reuse samples already generated for an unchanged initial path."""
    path = np.asarray(path, dtype=float)
    cached_path = plan.get("initial_motion_paths", {}).get(key)
    cached_samples = plan.get("initial_motion_samples", {}).get(key)
    settings = plan.get("initial_motion_settings")
    if (cached_path is not None and cached_samples is not None
            and settings == (float(vel), float(acc), float(SAMPLE_PERIOD))
            and path.shape == np.asarray(cached_path).shape
            and np.allclose(path, cached_path, atol=1e-6, rtol=0)):
        return np.asarray(cached_samples, dtype=float).copy()
    return timed_samples(path, vel, acc)


def concatenate_joint_paths(*paths):
    """Join joint paths at shared endpoints without duplicating waypoints."""
    joined = None
    for path in paths:
        points = np.asarray(path, dtype=float)
        if (points.ndim != 2 or points.shape[1] != 6 or len(points) == 0
                or not np.isfinite(points).all()):
            raise ValueError("Each joint path must be a non-empty finite Nx6 array.")
        if joined is None:
            joined = points.copy()
            continue
        if not np.allclose(joined[-1], points[0], atol=1e-6, rtol=0):
            raise ValueError("Joint paths do not meet at the same configuration.")
        joined = np.vstack((joined, points[1:]))
    return joined


def build_grasp_approach_path(plan, start_joints=None):
    """Keep every straight-approach waypoint, then append descent."""
    transfer = np.asarray(plan["pregrasp_path"], dtype=float).copy()
    if start_joints is not None:
        transfer[0] = np.asarray(start_joints, dtype=float)
    return concatenate_joint_paths(transfer, plan["descent"])


def merge_followup_actions(actions, vel, acc):
    """Time-parameterize selected consecutive actions as uninterrupted paths."""
    merge_groups = ((11, 12), (17, 18, 19), (20, 21),
                    (22, 23), (26, 27))
    merged, index = [], 0
    while index < len(actions):
        group = next((steps for steps in merge_groups
                      if index + len(steps) <= len(actions)
                      and tuple(item["step"] for item in
                                actions[index:index + len(steps)]) == steps), None)
        if group is None:
            merged.append(actions[index])
            index += 1
            continue

        source = actions[index:index + len(group)]
        item = dict(source[0])
        item["steps"] = group
        item["label"] = "连续执行参考步骤 {}-{}".format(group[0], group[-1])
        item["path"] = concatenate_joint_paths(*(part["path"] for part in source))
        item["samples"] = timed_samples(item["path"], vel, acc)
        item["target"] = source[-1]["target"]
        item["event"] = source[-1]["event"]
        if group == (20, 21):
            item["sync_after_step"] = 20
            item["sync_conf"] = np.asarray(source[0]["path"][-1], dtype=float).copy()
            item["sync_sample_index"] = int(np.argmin(np.max(
                np.abs(item["samples"] - item["sync_conf"]), axis=1)))
        item["obstacles"] = []
        for part in source:
            for obstacle in part.get("obstacles", []):
                if all(obstacle is not existing for existing in item["obstacles"]):
                    item["obstacles"].append(obstacle)
        merged.append(item)
        index += len(group)
    return merged


def synchronize_samples(sample_paths):
    """Pad shorter 5 ms sample streams at their endpoint for paired motion."""
    arrays = [np.asarray(path, dtype=float) for path in sample_paths]
    if not arrays or any(path.ndim != 2 or path.shape[1] != 6 or len(path) == 0
                         or not np.isfinite(path).all() for path in arrays):
        raise ValueError("Synchronized paths must be non-empty finite Nx6 arrays")
    max_length = max(len(path) for path in arrays)
    return [np.concatenate((path, np.repeat(path[-1:], max_length - len(path), axis=0)))
            for path in arrays]


def check_simultaneous_collision(robot, target, left_action, right_action):
    left_samples, right_samples = synchronize_samples(
        [left_action["samples"], right_action["samples"]]
    )
    obstacles = left_action.get("obstacles", [])
    for index, (left_q, right_q) in enumerate(zip(left_samples, right_samples)):
        if robot.fk("lft_arm", left_q) != "succ" or robot.fk("rgt_arm", right_q) != "succ":
            raise RuntimeError(f"步骤 8/9 同步路径第 {index} 点关节超限。")
        robot.get_oih_list()
        if collides_with_obstacles(robot, obstacles):
            raise RuntimeError(
                f"步骤 8/9 同步运动第 {index}/{len(left_samples) - 1} 点发生碰撞："
                f"{robot.last_collision_pairs}"
            )
        held = any(info["collision_model"] is target for info in robot.lft_oih_infos)
        if not held and robot.is_object_collided(target, contact_hand="lft_arm"):
            raise RuntimeError(
                f"步骤 8/9 同步运动第 {index}/{len(left_samples) - 1} 点与培养瓶碰撞："
                f"{robot.last_collision_pairs}"
            )


def reset_scene(plan):
    robot, target, tool = plan["robot"], plan["target"], plan["pipette"]
    robot.release("lft_arm", target)
    robot.release("rgt_arm", tool)
    robot.fk("lft_arm", plan["start_joints"])
    robot.fk("rgt_arm", plan["right_start"])
    robot.jaw_to("lft_arm", .0)
    target.set_homomat(plan["object_pose"])
    tool.set_homomat(plan["pipette_initial_pose"])
    robot.hold("rgt_arm", tool, TOOL_WIDTH)


def check_grasp_transfers(plan):
    """Recheck the high transfer and approach with the right arm and held tool."""
    robot, target = plan["robot"], plan["target"]
    obstacles = [plan["table_obstacle"]]
    reset_scene(plan)
    previous = plan["start_joints"]
    for destination in plan["hover_path"][1:]:
        count = max(2, int(np.ceil(np.max(np.abs(destination - previous)) / .02)) + 1)
        for q in np.linspace(previous, destination, count):
            check_configuration(robot, target, "lft_arm", q,
                                "高位 MoveJ（右手持枪）", [])
        previous = destination
    robot.jaw_to("lft_arm", .10)
    for q in plan["right_clearance"]["path"]:
        check_configuration(robot, target, "rgt_arm", q,
                            "打开左夹爪后的右臂让位", obstacles)
    for first, second in zip(plan["pregrasp_path"][:-1], plan["pregrasp_path"][1:]):
        count = max(2, int(np.ceil(np.max(np.abs(second - first)) / .02)) + 1)
        for q in np.linspace(first, second, count):
            check_configuration(robot, target, "lft_arm", q,
                                "直线到预抓取点（右手持枪）", obstacles)
    for q in plan["descent"]:
        check_configuration(robot, target, "lft_arm", q,
                            "下探（右手持枪）", obstacles)


def _reverse_opener_ascent(robot, target, descent, ascent):
    if (ascent["step"], descent["step"]) not in ((6, 7), (28, 29)):
        raise ValueError("只允许复用对应的开瓶上升路径。")
    path = np.asarray(ascent["path"], dtype=float)[::-1].copy()
    start = np.asarray(robot.get_jnt_values("lft_arm"), dtype=float)
    if not np.allclose(start, path[0], atol=1e-6, rtol=0):
        raise RuntimeError(descent["label"] + "：当前关节角与上升终点不一致。")

    start_tcp = tcp_pose(robot, "lft_arm")
    goal = descent["target"]
    delta = goal[:3, 3] - start_tcp[:3, 3]
    length2 = float(delta @ delta)
    for q in path:
        check_configuration(robot, target, "lft_arm", q, descent["label"],
                            descent.get("obstacles"),
                            descent.get("ignore_obstacle_pairs", ()))
        actual = tcp_pose(robot, "lft_arm")
        fraction = (float((actual[:3, 3] - start_tcp[:3, 3]) @ delta) / length2
                    if length2 > 1e-12 else 0.)
        nearest = start_tcp[:3, 3] + np.clip(fraction, 0., 1.) * delta
        if (np.linalg.norm(actual[:3, 3] - nearest) > .001
                or rotation_error(actual[:3, :3], goal[:3, :3]) > .002):
            raise RuntimeError(descent["label"] + "：倒序路径不符合原下降直线或姿态。")
    if (np.linalg.norm(actual[:3, 3] - goal[:3, 3]) > .001
            or rotation_error(actual[:3, :3], goal[:3, :3]) > .002):
        raise RuntimeError(descent["label"] + "：倒序路径终点与原目标不一致。")
    return path


def _check_timed_action_samples(robot, target, action, parts):
    samples = np.asarray(action["samples"], dtype=float)
    boundaries = [0]
    for part in parts[:-1]:
        offset = boundaries[-1]
        distance = np.max(np.abs(samples[offset:] - part["path"][-1]), axis=1)
        boundaries.append(offset + int(np.argmin(distance)))
    boundaries.append(len(samples) - 1)
    for part, first, last in zip(parts, boundaries[:-1], boundaries[1:]):
        if part["kind"] != "linear":
            continue
        for q in samples[max(0, first - 1):min(len(samples), last + 2)]:
            check_configuration(robot, target, part["arm"], q, part["label"],
                                part.get("obstacles"),
                                part.get("ignore_obstacle_pairs", ()))


def append_return_home_actions(plan, followup, step_paths, vel, acc,
                               left_home=LEFT_HOME, right_home=RIGHT_HOME):
    """Reuse the final grasp/return geometry, then move each arm to its home joints."""
    robot, target = plan["robot"], plan["target"]
    required = (21, 26, 27, 29)
    if any(step not in step_paths for step in required):
        raise RuntimeError("缺少结束回位所需的步骤 21、26、27 或 29 路径。")

    left_return = np.asarray(step_paths[27], dtype=float)[::-1].copy()
    if not np.allclose(robot.get_jnt_values("lft_arm"), left_return[0],
                       atol=1e-6, rtol=0):
        raise RuntimeError("结束回位 RRTC 起点与步骤 29 终点不连续。")
    descent = np.asarray(step_paths[26], dtype=float)[::-1].copy()
    lift = np.asarray(step_paths[26], dtype=float).copy()
    if not np.allclose(left_return[-1], descent[0], atol=1e-6, rtol=0):
        raise RuntimeError("结束回位 RRTC 路径与下探路径不连续。")
    if not np.allclose(descent[-1], lift[0], atol=1e-6, rtol=0):
        raise RuntimeError("结束回位下探与抬升路径不连续。")

    def append_path(step, arm, path, label, event=None, ignore=()):
        path = np.asarray(path, dtype=float)
        for q in path:
            check_configuration(robot, target, arm, q, label, [], ignore)
        samples = timed_samples(path, vel, acc)
        action = dict(step=step, arm=arm, kind="path", path=path,
                      samples=samples, label=label, event=event,
                      obstacles=[], ignore_obstacle_pairs=tuple(ignore))
        followup.append(action)
        robot.fk(arm, path[-1])
        robot.get_oih_list()
        apply_event(robot, target, event)
        return path

    append_path(30, "lft_arm", left_return,
                "反向复用步骤 27 的 RRTC 路径返回瓶子上方")
    append_path(31, "lft_arm", descent,
                "复用步骤 26 的反向路径下探并放回瓶子", "release",
                [("lft_arm/held_object", 0)])
    append_path(32, "lft_arm", lift, "复用步骤 26 的路径抬起空夹爪")

    right_path = np.vstack((robot.get_jnt_values("rgt_arm"),
                            np.asarray(right_home, dtype=float)))
    append_path(33, "rgt_arm", right_path, "右臂直接返回固定初始关节角")
    followup[-1]["parallel_group"] = "return_home"
    left_path = np.vstack((robot.get_jnt_values("lft_arm"),
                           np.asarray(left_home, dtype=float)))
    append_path(34, "lft_arm", left_path, "左臂直接返回固定初始关节角")
    followup[-1]["parallel_group"] = "return_home"
    return followup


def build_followup_plan(plan, vel=.3, acc=1., check_opener_collisions=True,
                        opener_target_world_x_offset=0.0,
                        validate_grasp_transfer=True,
                        optimize_merged_paths=False):
    robot, target = plan["robot"], plan["target"]
    if not optimize_merged_paths:
        print("规划后续动作：沿用参考设备位置，使用当前模型、cf.stl 和桌面碰撞网格。", flush=True)
        print("右手须预先持有移液枪；本流程只执行机械臂和左夹爪动作。", flush=True)
    actions = station_actions(plan, check_opener_collisions=check_opener_collisions)
    if opener_target_world_x_offset:
        for action in actions:
            if action["step"] in (5, 6, 7, 27, 28, 29):
                action["target"] = action["target"].copy()
                action["target"][0, 3] += float(opener_target_world_x_offset)
    try:
        if validate_grasp_transfer:
            check_grasp_transfers(plan)
        robot.fk("lft_arm", plan["descent"][-1])
        target.set_homomat(plan["object_pose"])
        apply_event(robot, target, "hold")
        if validate_grasp_transfer:
            for q in plan["descent"][::-1]:
                check_configuration(robot, target, "lft_arm", q,
                                    "夹住后返回预抓取点")
        by_step = {}
        for action in actions:
            step = action["step"]
            if optimize_merged_paths and step in (7, 29):
                path = _reverse_opener_ascent(
                    robot, target, action, by_step[step - 1])
            else:
                if not optimize_merged_paths:
                    print(f"规划参考步骤 {step}/29：{action['label']}", flush=True)
                path = solve_path(robot, target, action)
            action["path"] = path
            by_step[step] = action
            if (not optimize_merged_paths
                    and action["kind"] in ("linear", "path")):
                action["samples"] = timed_samples(path, vel, acc)
                for q in action["samples"]:
                    check_configuration(robot, target, action["arm"], q,
                                        action["label"], action.get("obstacles"),
                                        action.get("ignore_obstacle_pairs", ()))
            robot.fk(action["arm"], path[-1])
            robot.get_oih_list()
            apply_event(robot, target, action["event"])

        followup = merge_followup_actions(actions, vel, acc)
        if optimize_merged_paths:
            for action in followup:
                if action["step"] == 5:
                    continue
                parts = [by_step[step] for step in
                         action.get("steps", (action["step"],))]
                if "samples" not in action:
                    action["samples"] = timed_samples(action["path"], vel, acc)
                _check_timed_action_samples(robot, target, action, parts)

        step8 = next(action for action in followup if action["step"] == 8)
        step9 = next(action for action in followup if action["step"] == 9)
        robot.fk("lft_arm", step8["samples"][0])
        robot.fk("rgt_arm", step9["samples"][0])
        check_simultaneous_collision(robot, target, step8, step9)

        if optimize_merged_paths:
            step_paths = {step: np.asarray(action["path"], dtype=float).copy()
                          for step, action in by_step.items()}
            plan["followup_step_paths"] = step_paths
            robot.fk("lft_arm", step_paths[29][-1])
            robot.fk("rgt_arm", step_paths[21][-1])
            robot.get_oih_list()
            append_return_home_actions(plan, followup, step_paths, vel, acc)

        plan["followup"] = followup
        if (optimize_merged_paths and followup
                and followup[0]["step"] == 5):
            plan["lift_and_step5_path"] = concatenate_joint_paths(
                np.asarray(plan["descent"], dtype=float)[::-1],
                followup[0]["path"])
            plan["lift_and_step5_samples"] = timed_samples(
                plan["lift_and_step5_path"], vel, acc)
    finally:
        reset_scene(plan)
    return plan


def wait_at_target(arm, joints, label, timeout=10.):
    """Do not start the next arm/event merely because a path RPC returned."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        actual = np.asarray(arm.get_jnt_values())
        status = arm.get_motion_status()
        arm._check_motion_ready(status)
        if (actual.shape == (6,) and np.isfinite(actual).all()
                and np.max(np.abs(actual - joints)) <= .01 and status["steady"] is True):
            return
        time.sleep(.02)
    raise RuntimeError(label + "：未确认实际到位，停止后续流程。")


def wait_at_joint_waypoint(arm, start_joints, target_joints, label,
                           motion_future=None, timeout=60., tolerance=.02):
    """Detect an intermediate path waypoint while the controller keeps moving."""
    deadline = time.monotonic() + timeout
    start = np.asarray(start_joints, dtype=float)
    target = np.asarray(target_joints, dtype=float)
    motion_started = False
    while time.monotonic() < deadline:
        actual = np.asarray(arm.get_jnt_values(), dtype=float)
        status = arm.get_motion_status()
        arm._check_motion_ready(status)
        if actual.shape != (6,) or not np.isfinite(actual).all():
            raise RuntimeError(label + "：读取到无效关节角。")
        if np.max(np.abs(actual - start)) > .02:
            motion_started = True
        if motion_started and np.max(np.abs(actual - target)) <= tolerance:
            return
        if motion_future is not None and motion_future.done():
            motion_future.result()
            raise RuntimeError(label + "：右臂已完成整条路径，但未检测到同步路点。")
        time.sleep(.005)
    raise TimeoutError(label + "：等待关节路径同步路点超时。")


def execute_right_clearance(plan, left, right, vel=.3, acc=1.):
    """Finish the right-arm MoveJ before allowing the left arm to descend."""
    try:
        wait_at_target(left, plan["hover_path"][-1], "左臂高位等待点")
        wait_at_target(right, plan["right_start"], "右臂让位起点")
        right.move_jntspace_path(
            np.asarray([right.get_jnt_values(), plan["right_wait"]]),
            vel=vel, acc=acc, execution_mode="buffer")
        wait_at_target(right, plan["right_wait"], "右臂让位终点")
    except Exception:
        for arm in (left, right):
            try:
                arm.stop()
            except Exception:
                pass
        raise


def execute_parallel_actions(actions, arms, target, gripper, vel, acc):
    if (len(actions) != 2
            or {action["arm"] for action in actions} != {"lft_arm", "rgt_arm"}):
        raise ValueError("Concurrent motion requires one left-arm and one right-arm path.")
    buffers = {}
    try:
        for action in actions:
            arm = arms[action["arm"]]
            wait_at_target(arm, action["path"][0], action["label"] + " start")

        for action in actions:
            arm = arms[action["arm"]]
            buffers[action["arm"]] = arm.prepare_jntspace_path_buffer(
                action["samples"], control_frequency=SAMPLE_PERIOD,
                vel=vel, acc=acc,
            )

        start_barrier = Barrier(len(actions))

        def run_action(action):
            start_barrier.wait(timeout=10.)
            arms[action["arm"]].move_jntspace_path(
                action["path"], vel=vel, acc=acc,
                control_frequency=SAMPLE_PERIOD,
                execution_mode="buffer",
                prepared_samples=action["samples"],
                prepared_buffer=buffers[action["arm"]],
                motion_idle_verified=True,
            )

        failures = []
        with ThreadPoolExecutor(max_workers=len(actions)) as executor:
            futures = [executor.submit(run_action, action) for action in actions]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    failures.append(exc)
                    for arm in arms.values():
                        try:
                            arm.stop()
                        except Exception:
                            pass
        if failures:
            raise RuntimeError("Concurrent motion failed: {}".format(
                failures[0]
            )) from failures[0]

        for action in actions:
            arm = arms[action["arm"]]
            wait_at_target(arm, action["path"][-1], action["label"])
        for action in actions:
            if action["event"] == "open":
                gripper.open_g()
            elif action["event"] == "release":
                gripper.open_g()
            elif action["event"] == "hold":
                gripper.close_g()
    except Exception:
        for arm in arms.values():
            try:
                arm.stop()
            except Exception:
                pass
        raise
    finally:
        for action in actions:
            arm = arms[action["arm"]]
            buffer = buffers.get(action["arm"])
            if buffer is not None and buffer.get("state") == "prepared":
                try:
                    arm.release_prepared_jntspace_path_buffer(buffer)
                except Exception:
                    pass


def execute_parallel_with_left_continuation(actions, continuation, arms,
                                             target, gripper, vel, acc):
    """Start left steps 22-24 as the right 20-21 path crosses step 20."""
    if (len(actions) != 2
            or {action["arm"] for action in actions} != {"lft_arm", "rgt_arm"}
            or continuation["arm"] != "lft_arm"
            or tuple(next(action for action in actions
                          if action["arm"] == "rgt_arm").get("steps", ())) != (20, 21)):
        raise ValueError("Expected right/left parallel actions and a left continuation.")
    buffers = {}

    def prepare(action):
        arm = arms[action["arm"]]
        buffers[id(action)] = arm.prepare_jntspace_path_buffer(
            action["samples"], control_frequency=SAMPLE_PERIOD,
            vel=vel, acc=acc,
        )

    def run(action):
        arm = arms[action["arm"]]
        arm.move_jntspace_path(
            action["path"], vel=vel, acc=acc,
            control_frequency=SAMPLE_PERIOD,
            execution_mode="buffer", prepared_samples=action["samples"],
            prepared_buffer=buffers[id(action)], motion_idle_verified=True,
        )

    try:
        for action in actions:
            arm = arms[action["arm"]]
            wait_at_target(arm, action["path"][0], action["label"] + " start")
            prepare(action)

        with ThreadPoolExecutor(max_workers=3) as executor:
            right_action = next(action for action in actions
                                if action["arm"] == "rgt_arm")
            left_action = next(action for action in actions
                               if action["arm"] == "lft_arm")
            if "sync_conf" not in right_action:
                raise ValueError("Merged step 20-21 path is missing its step 20 waypoint.")
            try:
                right_future = executor.submit(run, right_action)
                waypoint_timeout = max(
                    10., 2. * right_action.get("sync_sample_index", 0)
                    * SAMPLE_PERIOD + 5.)
                wait_at_joint_waypoint(
                    arms["rgt_arm"], right_action["path"][0],
                    right_action["sync_conf"], "步骤 20/21 同步路点",
                    motion_future=right_future, timeout=waypoint_timeout)
                left_future = executor.submit(run, left_action)

                # Step 24 starts after step 23 releases the bottle; the right arm
                # continues step 21 in the background.
                left_future.result()
                wait_at_target(arms["lft_arm"], left_action["path"][-1],
                               left_action["label"])
                if left_action["event"] == "release":
                    gripper.open_g()
                elif left_action["event"] == "hold":
                    gripper.close_g()

                wait_at_target(arms["lft_arm"], continuation["path"][0],
                               continuation["label"] + " start")
                prepare(continuation)
                continuation_future = executor.submit(run, continuation)
                right_future.result()
                continuation_future.result()
            except Exception:
                for arm in arms.values():
                    try:
                        arm.stop()
                    except Exception:
                        pass
                raise

        wait_at_target(arms["rgt_arm"], right_action["path"][-1],
                       right_action["label"])
        wait_at_target(arms["lft_arm"], continuation["path"][-1],
                       continuation["label"])
        if continuation["event"] == "release":
            gripper.open_g()
        elif continuation["event"] == "hold":
            gripper.close_g()
    except Exception:
        for arm in arms.values():
            try:
                arm.stop()
            except Exception:
                pass
        raise
    finally:
        for action in (*actions, continuation):
            arm = arms[action["arm"]]
            buffer = buffers.get(id(action))
            if buffer is not None and buffer.get("state") == "prepared":
                try:
                    arm.release_prepared_jntspace_path_buffer(buffer)
                except Exception:
                    pass


def execute_followup(plan, left, right, gripper, vel=.3, acc=1.):
    arms = {"lft_arm": left, "rgt_arm": right}
    try:
        wait_at_target(left, plan["descent"][-1], "夹住物体后的抓取点")
        wait_at_target(right, plan["right_wait"], "右臂让位后的等待位置")
        actions = plan["followup"]
        index = 0
        if actions and actions[0]["step"] == 5:
            lift_and_step5 = plan.get("lift_and_step5_path")
            if lift_and_step5 is None:
                lift_and_step5 = concatenate_joint_paths(
                    np.asarray(plan["descent"])[::-1], actions[0]["path"])
            samples = plan.get("lift_and_step5_samples")
            if samples is None:
                samples = timed_samples(lift_and_step5, vel, acc)
            action = actions[0]
            print("夹持后抬升，并连续执行步骤 5 到开瓶器位置。", flush=True)
            wait_at_target(left, lift_and_step5[0], "抬升及步骤 5 起点")
            left.move_jntspace_path(
                lift_and_step5, vel=vel, acc=acc,
                control_frequency=SAMPLE_PERIOD,
                execution_mode="buffer", prepared_samples=samples)
            wait_at_target(left, lift_and_step5[-1], action["label"])
            index = 1
        while index < len(actions):
            action = actions[index]
            if action.get("parallel_group") == "return_home":
                if (index + 1 >= len(actions)
                        or actions[index + 1].get("parallel_group") != "return_home"
                        or {action["arm"], actions[index + 1]["arm"]}
                        != {"lft_arm", "rgt_arm"}):
                    raise RuntimeError("左右臂回位动作未组成有效的同步组。")
                paired = actions[index:index + 2]
                synchronized = synchronize_samples(
                    [item["samples"] for item in paired])
                paired = [dict(item, samples=samples)
                          for item, samples in zip(paired, synchronized)]
                print("步骤 33-34：左右臂同时返回初始关节角。", flush=True)
                execute_parallel_actions(paired, arms, plan.get("target"),
                                         gripper, vel, acc)
                index += 2
                continue
            if (action["step"] == 8 and index + 1 < len(actions)
                    and actions[index + 1]["step"] == 9):
                paired = actions[index:index + 2]
                print("步骤 8 左臂与步骤 9 右臂同时运动。", flush=True)
                execute_parallel_actions(paired, arms, plan.get("target"), gripper,
                                         vel, acc)
                index += 2
                continue
            if (tuple(action.get("steps", (action["step"],))) == (20, 21)
                    and index + 1 < len(actions)
                    and tuple(actions[index + 1].get(
                        "steps", (actions[index + 1]["step"],))) == (22, 23)):
                paired = actions[index:index + 2]
                if (index + 2 < len(actions)
                        and actions[index + 2]["step"] == 24):
                    print("步骤 20-21 连续运动；到达步骤 20 终点时，左臂启动 22-23，松爪后接 24。",
                          flush=True)
                    execute_parallel_with_left_continuation(
                        paired, actions[index + 2], arms, plan.get("target"),
                        gripper, vel, acc)
                    index += 3
                else:
                    print("步骤 20-21 右臂与步骤 22-23 左臂同时运动。", flush=True)
                    execute_parallel_actions(paired, arms, plan.get("target"),
                                             gripper, vel, acc)
                    index += 2
                continue
            arm, path = arms[action["arm"]], action["path"]
            steps = action.get("steps", (action["step"],))
            step_label = (f"{steps[0]}-{steps[-1]}"
                          if len(steps) > 1 else str(steps[0]))
            total_steps = 34 if action["step"] >= 30 else 29
            print(f"后续步骤 {step_label}/{total_steps}：{action['label']}", flush=True)
            wait_at_target(arm, path[0], action["label"] + "起点")
            arm.move_jntspace_path(path, vel=vel, acc=acc, execution_mode="buffer",
                                   prepared_samples=action["samples"])
            wait_at_target(arm, path[-1], action["label"])
            if action["event"] == "release":
                gripper.open_g()
            elif action["event"] == "hold":
                gripper.close_g()
            index += 1
    except Exception:
        for arm in arms.values():
            try:
                arm.stop()
            except Exception:
                pass
        raise
