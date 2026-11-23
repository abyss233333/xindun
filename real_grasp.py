"""Capture a bottle pose, grasp it, then run the dual-arm workcell sequence.

Default mode captures a fresh camera pose and plans from configured start
joints without connecting to robot hardware. Add ``--execute`` to read both
real arms and execute after the entire plan succeeds. The right hand must
already hold the pipette. Use preview_live_workflow.py for the camera animation.
"""

import argparse
from contextlib import contextmanager
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime
from time import perf_counter
from unittest.mock import patch

import numpy as np


SCENE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.abspath(os.path.join(SCENE_DIR, ".."))
for _path in (SCENE_DIR, REPO_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from robot_sim.robots.i5.i5_dual_427_2 import I5Dual
import modeling.collision_model as cm
import motion.probabilistic.rrt_connect as rrtc
import basis.robot_math as rm


POSE_RESULT_ROOT = os.path.join(
    r"D:\FoundationPose-main\FoundationPose-main",
    "foundationpose_world_result",
)
TABLE_REFERENCE_POSE_FILE = os.path.join(
    POSE_RESULT_ROOT, "20261116_203145_482154", "pose.json")
FOUNDATIONPOSE_ROOT = Path(r"D:\FoundationPose-main\FoundationPose-main")
FOUNDATIONPOSE_SCRIPT = FOUNDATIONPOSE_ROOT / "foundationpose_d435_world_visualize.py"
FOUNDATIONPOSE_PYTHON = Path(r"D:\conda-envs\foundationpose-win\python.exe")
FOUNDATIONPOSE_MESH = Path(REPO_DIR) / "427" / "meshes" / "cf.STL"
SAM_CHECKPOINT = Path(r"D:\sam_seg\SAM-Tool-main\sam_vit_h_4b8939.pth")
REFERENCE_ORIENTATION_POSE = Path(os.environ.get(
    "XINDUN_REFERENCE_ORIENTATION_POSE",
    r"D:\FoundationPose-main\FoundationPose-main\foundationpose_world_result\20261113_220757_649151\pose.json",
))
LEFT_ROBOT_IP = "192.168.110.12"
RIGHT_ROBOT_IP = "192.168.110.11"
ROBOT_PORT = 30004
GRIPPER_PORT = "com10"
GRIPPER_ID = 2
GRIPPER_BAUDRATE = 115200
GRIPPER_READ_TIMEOUT = 1.0
MOVE_VEL = 0.8
MOVE_ACC = 1.0
LFT_OPEN_WIDTH = 0.10
PREGRASP_CLEARANCE = 0.06
HIGH_APPROACH = 0.30
DESCENT_STEP = 0.001

# Compensate +10 mm along object-local Y and +5 mm along object-local Z.
POSE_COMPENSATION_OBJECT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.010],
    [0.0, 0.0, 1.0, 0.005],
    [0.0, 0.0, 0.0, 1.0],
])
TABLE_REFERENCE_COMPENSATION_OBJECT = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.010],
    [0.0, 0.0, 1.0, 0.005],
    [0.0, 0.0, 0.0, 1.0],
])

# Same configured start pose used by the simulation and the existing real.py.
CONFIGURED_START = np.array([
    -5.57630884 + 2 * math.pi,
    -5.99696722 + 2 * math.pi,
    -3.9449993 + 2 * math.pi,
    -5.38125836 + 2 * math.pi,
    -3.721296 + 2 * math.pi,
    -0.12433301,
])


@contextmanager
def _collision_check_scope(enabled):
    previous = enabled[0]
    enabled[0] = True
    try:
        yield
    finally:
        enabled[0] = previous


@contextmanager
def _selective_collision_policy():
    """Check only the initial paired move and left-arm RRTC paths."""
    import workcell_followup

    enabled = [False]
    original_check = workcell_followup.check_configuration
    original_collides = workcell_followup.collides_with_obstacles
    original_simultaneous = workcell_followup.check_simultaneous_collision
    original_solve_path = workcell_followup.solve_path
    original_object_collision = I5Dual.is_object_collided

    def check_configuration(robot, target, arm, q, label, obstacles=None,
                            ignore_obstacle_pairs=()):
        if enabled[0]:
            return original_check(robot, target, arm, q, label, obstacles,
                                  ignore_obstacle_pairs)
        if robot.fk(arm, q) != "succ":
            raise RuntimeError(label + "：关节超出范围。")
        robot.get_oih_list()

    def collides_with_obstacles(robot, obstacles, otherrobot_list=None,
                                ignore_obstacle_pairs=()):
        if not enabled[0]:
            return False
        return original_collides(robot, obstacles, otherrobot_list,
                                 ignore_obstacle_pairs)

    def is_object_collided(robot, *args, **kwargs):
        if not enabled[0]:
            return False
        return original_object_collision(robot, *args, **kwargs)

    def check_simultaneous_collision(robot, target, left_action, right_action):
        if (left_action.get("label") != "左臂到高位点"
                or right_action.get("label") != "右臂让位"):
            return None
        with _collision_check_scope(enabled):
            return original_simultaneous(robot, target, left_action, right_action)

    def solve_path(robot, target, action):
        if (action.get("arm") == "lft_arm"
                and action.get("planner") == "rrtc"):
            with _collision_check_scope(enabled):
                return original_solve_path(robot, target, action)
        return original_solve_path(robot, target, action)

    with patch.object(workcell_followup, "check_configuration",
                      side_effect=check_configuration), \
            patch.object(workcell_followup, "collides_with_obstacles",
                         side_effect=collides_with_obstacles), \
            patch.object(workcell_followup, "check_simultaneous_collision",
                         side_effect=check_simultaneous_collision), \
            patch.object(workcell_followup, "solve_path",
                         side_effect=solve_path), \
            patch.object(I5Dual, "is_object_collided", is_object_collided):
        yield


def load_reference_orientation(pose_file):
    """Load only the reference object's world orientation from a pose sample."""
    with open(pose_file, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    reference_pose = np.asarray(payload.get("T_world_object"), dtype=float)
    if reference_pose.shape != (4, 4) or not np.isfinite(reference_pose).all():
        raise ValueError("参考姿态文件中的 T_world_object 必须是有限的 4x4 矩阵。")
    if not np.allclose(reference_pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("参考姿态文件中的 T_world_object 不是合法齐次变换矩阵。")
    reference_rot = reference_pose[:3, :3]
    if (not np.allclose(reference_rot.T @ reference_rot, np.eye(3), atol=1e-3)
            or np.linalg.det(reference_rot) < 0.99):
        raise ValueError("参考姿态文件中的旋转部分不是合法旋转矩阵。")
    return reference_rot


def load_table_reference_pose():
    """Return the compensated reference pose used to place the fixed table."""
    with open(TABLE_REFERENCE_POSE_FILE, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    table_pose = np.asarray(payload.get("T_world_object"), dtype=float)
    if (table_pose.shape != (4, 4) or not np.isfinite(table_pose).all()
            or not np.allclose(table_pose[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("桌面参考 pose.json 中的 T_world_object 无效。")
    table_pose[:3, :3] = load_reference_orientation(REFERENCE_ORIENTATION_POSE)
    return table_pose @ TABLE_REFERENCE_COMPENSATION_OBJECT


def capture_live_pose():
    """Run the D435/SAM/FoundationPose one-frame registration for this grasp."""
    required_paths = (FOUNDATIONPOSE_SCRIPT, FOUNDATIONPOSE_PYTHON,
                      FOUNDATIONPOSE_MESH, SAM_CHECKPOINT,
                      REFERENCE_ORIENTATION_POSE)
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "现场位姿识别所需文件不存在：\n  " + "\n  ".join(missing))

    os.makedirs(POSE_RESULT_ROOT, exist_ok=True)
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(POSE_RESULT_ROOT, run_name)
    os.makedirs(output_dir, exist_ok=False)
    pose_file = os.path.join(output_dir, "pose.json")

    command = [
        str(FOUNDATIONPOSE_PYTHON), str(FOUNDATIONPOSE_SCRIPT),
        "--estimate-only",
        "--mesh-file", str(FOUNDATIONPOSE_MESH),
        "--sam-checkpoint", str(SAM_CHECKPOINT),
        "--reference-orientation-pose", str(REFERENCE_ORIENTATION_POSE),
        "--pose-file", pose_file,
        "--output-dir", output_dir,
    ]
    print("开始 D435 现场识别。请在 SAM 窗口点选目标物块并确认。")
    subprocess.run(command, cwd=str(FOUNDATIONPOSE_ROOT), check=True)
    if not os.path.isfile(pose_file):
        raise RuntimeError("现场识别进程结束，但没有生成本次 pose.json。")

    return load_grasp_pose(pose_file)


def load_grasp_pose(pose_file):
    """Preserve the detected object orientation and apply local compensation."""
    with open(pose_file, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    detected_pose = np.asarray(payload.get("T_world_object"), dtype=float)
    if detected_pose.shape != (4, 4) or not np.isfinite(detected_pose).all():
        raise ValueError("pose.json 中的 T_world_object 必须是有限的 4x4 矩阵。")
    if not np.allclose(detected_pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("T_world_object 不是合法齐次变换矩阵。")
    if (not np.allclose(detected_pose[:3, :3].T @ detected_pose[:3, :3],
                        np.eye(3), atol=1e-3)
            or np.linalg.det(detected_pose[:3, :3]) < 0.99):
        raise ValueError("T_world_object 的旋转部分不是合法旋转矩阵。")
    detected_rotation = detected_pose[:3, :3].copy()
    reference_z = load_reference_orientation(
        REFERENCE_ORIENTATION_POSE)[:, 2]
    selected_z = detected_rotation[:, 2]
    z_axis_error_deg = float(np.degrees(np.arccos(np.clip(
        np.dot(selected_z, reference_z), -1.0, 1.0))))
    x_axis = detected_rotation[:, 0] - reference_z * np.dot(
        detected_rotation[:, 0], reference_z)
    if np.linalg.norm(x_axis) < 1e-9:
        y_axis = detected_rotation[:, 1] - reference_z * np.dot(
            detected_rotation[:, 1], reference_z)
        if np.linalg.norm(y_axis) < 1e-9:
            raise ValueError("识别姿态无法确定绕物体 Z 轴的旋转角。")
        x_axis = np.cross(y_axis, reference_z)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(reference_z, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    detected_rotation = np.column_stack((x_axis, y_axis, reference_z))
    detected_pose[:3, :3] = detected_rotation
    pose = detected_pose @ POSE_COMPENSATION_OBJECT
    print("本次相机识别位姿: {}".format(pose_file))
    print("相机检测中心 (WRS base/world, m): {}".format(
        detected_pose[:3, 3].tolist()))
    print("物体局部坐标补偿矩阵 (Y 轴 +10 mm，Z 轴 +5 mm):\n{}".format(
        POSE_COMPENSATION_OBJECT))
    print("补偿后目标位置 (WRS base/world, m): {}".format(
        pose[:3, 3].tolist()))
    print("FoundationPose 候选 Z 轴与样本夹角: {:.2f} 度；已对齐 Z 轴，保留其绕 Z 轴的角度。".format(
        z_axis_error_deg))
    print("用于抓取规划的物体旋转矩阵:\n{}".format(detected_rotation))
    return pose, os.path.abspath(pose_file)


def load_target_mesh(position, rotation):
    mesh_path = os.path.join(SCENE_DIR, "objects", "cf.stl")
    if not os.path.isfile(mesh_path):
        mesh_path = os.path.join(SCENE_DIR, "objects", "cf.STL")
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError("目标网格不存在：objects/cf.stl")
    target = cm.CollisionModel(mesh_path)
    target.set_pos(position)
    target.set_rotmat(rotation)
    return target


def find_collision_free_ik(robot, position, rotation, seed, obstacles, label):
    from workcell_followup import collides_with_obstacles

    seeds = [np.asarray(seed, dtype=float)]
    rng = np.random.default_rng(20261113)
    seeds.extend(rng.uniform(-math.pi, math.pi, size=6) for _ in range(64))
    saved_conf = robot.get_jnt_values("lft_arm").copy()
    best_conf = None
    best_distance = float("inf")
    try:
        for candidate in seeds:
            conf = robot.tracik(
                tgt_pos=position,
                tgt_rotmat=rotation,
                component_name="lft_arm",
                solver_type="Distance",
                seed_jnt_values=candidate,
            )
            if conf is None:
                continue
            conf = np.asarray(conf, dtype=float).copy()
            conf += 2 * math.pi * np.round((seed - conf) / (2 * math.pi))
            if not robot.is_jnt_values_in_ranges("lft_arm", conf):
                continue
            robot.fk("lft_arm", conf)
            if collides_with_obstacles(robot, obstacles):
                continue
            distance = float(np.linalg.norm(conf - seed))
            if distance < best_distance:
                best_conf, best_distance = conf.copy(), distance
    finally:
        robot.fk("lft_arm", saved_conf)
    if best_conf is not None:
        return best_conf
    raise RuntimeError("{} 没有找到无碰撞的逆运动学解。".format(label))


def plan_rrtc(robot, start, goal, obstacles, label):
    from workcell_followup import _MeshObstacleCollisionView

    planner = rrtc.RRTConnect(_MeshObstacleCollisionView(robot))
    path = planner.plan(
        component_name="lft_arm",
        start_conf=start,
        goal_conf=goal,
        obstacle_list=obstacles,
        ext_dist=0.30,
        max_iter=12000,
        max_time=60.0,
        smoothing_iterations=80,
    )
    if path is None:
        raise RuntimeError("{} 的 RRTC 规划失败。".format(label))
    for conf in path[1:]:
        if robot.fk("lft_arm", conf) != "succ":
            raise RuntimeError("{} 路径包含越界关节配置。".format(label))
    return path


def build_grasp_plan(start_joints, pose, pose_file, right_start=None):
    robot = I5Dual()
    robot.lft_hnd.jawwidth_rng = [0.0, LFT_OPEN_WIDTH]
    for finger in (robot.lft_hnd.lft, robot.lft_hnd.rgt):
        finger.jnts[1]["motion_rng"] = [0.0, LFT_OPEN_WIDTH / 2.0]
    if right_start is not None:
        right_start = np.asarray(right_start, dtype=float)
        if (right_start.shape != (6,) or not np.isfinite(right_start).all()
                or robot.fk("rgt_arm", right_start) != "succ"):
            raise ValueError("右臂起始关节角无效。")
    right_start = robot.get_jnt_values("rgt_arm").copy()

    start_joints = np.asarray(start_joints, dtype=float).reshape(-1)
    if start_joints.shape != (6,) or not np.isfinite(start_joints).all():
        raise ValueError("左臂当前关节角必须是 6 个有限弧度值。")
    if not robot.is_jnt_values_in_ranges("lft_arm", start_joints):
        raise ValueError("左臂当前关节角超出 I5Dual 模型范围；停止规划。")
    if robot.fk("lft_arm", start_joints) != "succ":
        raise RuntimeError("无法将当前关节角载入 I5Dual 模型。")

    object_pos, object_rot = pose[:3, 3], pose[:3, :3]
    target = load_target_mesh(object_pos, object_rot)
    from workcell_followup import (SAMPLE_PERIOD,
                                   check_simultaneous_collision,
                                   collides_with_obstacles,
                                   make_table_obstacle, make_pipette,
                                   plan_right_clearance, solve_linear_point,
                                   solve_path, timed_samples)
    table_obstacle = make_table_obstacle(load_table_reference_pose(), target)
    obstacles = [target, table_obstacle]
    pipette = make_pipette(robot)
    pipette_initial_pose = pipette.get_homomat().copy()

    jaw_to_object = np.eye(4)
    jaw_to_object[:3, :3] = rm.rotmat_from_axangle([1, 0, 0], math.pi)
    jaw_to_object[:3, 3] = [0.0, 0.0, 0.025]
    jaw_to_world = pose @ jaw_to_object
    grasp_pos, grasp_rot = jaw_to_world[:3, 3], jaw_to_world[:3, :3]
    pregrasp_pos = grasp_pos + np.array([0.0, 0.0, PREGRASP_CLEARANCE])
    hover_pos = grasp_pos + np.array([0.0, 0.0, HIGH_APPROACH])

    hover_conf = find_collision_free_ik(
        robot, hover_pos, grasp_rot, start_joints, [target], "高位等待点")
    # 第一段高位移动按用户要求直接 MoveJ，不经过 RRTC。
    hover_path = np.asarray([start_joints.copy(), hover_conf.copy()])

    robot.jaw_to("lft_hnd", LFT_OPEN_WIDTH)
    robot.fk("lft_arm", hover_path[-1])
    right_clearance = plan_right_clearance(robot, target, [table_obstacle])
    right_wait = robot.get_jnt_values("rgt_arm").copy()
    initial_left_path = hover_path.copy()
    initial_right_path = np.asarray([right_start, right_wait])
    initial_left_samples = timed_samples(initial_left_path, MOVE_VEL, MOVE_ACC)
    initial_right_samples = timed_samples(initial_right_path, MOVE_VEL, MOVE_ACC)
    initial_left_action = dict(
        arm="lft_arm", label="左臂到高位点", obstacles=[],
        samples=initial_left_samples)
    initial_right_action = dict(
        arm="rgt_arm", label="右臂让位",
        samples=initial_right_samples)
    saved_left = robot.get_jnt_values("lft_arm").copy()
    saved_right = robot.get_jnt_values("rgt_arm").copy()
    saved_jaw_width = robot.lft_hnd.get_jawwidth()
    try:
        robot.jaw_to("lft_hnd", 0.0)
        robot.fk("lft_arm", start_joints)
        robot.fk("rgt_arm", right_start)
        check_simultaneous_collision(robot, target, initial_left_action,
                                     initial_right_action)
    finally:
        robot.fk("lft_arm", saved_left)
        robot.fk("rgt_arm", saved_right)
        robot.jaw_to("lft_hnd", saved_jaw_width)
        robot.get_oih_list()
    # Follow a straight TCP line from hover, continuing the same IK branch.
    robot.fk("lft_arm", hover_conf)
    pregrasp_path = solve_path(robot, target, dict(
        arm="lft_arm", kind="linear", label="高位点直线到预抓取点",
        target=rm.homomat_from_posrot(pregrasp_pos, grasp_rot),
        obstacles=obstacles))
    pregrasp_conf = pregrasp_path[-1].copy()

    descent_count = max(
        2, int(np.ceil(np.linalg.norm(pregrasp_pos - grasp_pos) / DESCENT_STEP)))
    descent = [pregrasp_conf]
    previous_conf = pregrasp_conf
    for index in range(1, descent_count + 1):
        target_pos = pregrasp_pos + (grasp_pos - pregrasp_pos) * (index / descent_count)
        conf, reason = solve_linear_point(
            robot, "lft_arm", target_pos, grasp_rot, previous_conf)
        if conf is None:
            raise RuntimeError("下降路径第 {} 步没有连续逆解：{}".format(index, reason))
        if robot.fk("lft_arm", conf) != "succ":
            raise RuntimeError("下降路径包含越界关节配置。")
        if collides_with_obstacles(robot, [table_obstacle]):
            raise RuntimeError("下降路径发生机器人自身碰撞。")
        if robot.is_object_collided(target):
            raise RuntimeError("下探时张开的夹爪或机械臂与目标物体碰撞。")
        descent.append(conf.copy())
        previous_conf = conf

    return {
        "robot": robot,
        "target": target,
        "table_obstacle": table_obstacle,
        "object_pose": pose.copy(),
        "jaw_to_object": jaw_to_object.copy(),
        "start_joints": start_joints.copy(),
        "right_start": right_start,
        "right_wait": right_wait,
        "right_clearance": right_clearance,
        "pipette": pipette,
        "pipette_initial_pose": pipette_initial_pose,
        "pose_file": pose_file,
        "object_pos": object_pos.copy(),
        "grasp_pos": grasp_pos.copy(),
        "pregrasp_pos": pregrasp_pos.copy(),
        "hover_pos": hover_pos.copy(),
        "hover_path": hover_path,
        "pregrasp_path": pregrasp_path,
        "descent": descent,
        "initial_motion_paths": {
            "left": initial_left_path.copy(),
            "right": initial_right_path.copy(),
        },
        "initial_motion_samples": {
            "left": initial_left_samples.copy(),
            "right": initial_right_samples.copy(),
        },
        "initial_motion_settings": (
            float(MOVE_VEL), float(MOVE_ACC),
            float(SAMPLE_PERIOD)),
    }


def get_real_gripper_class():
    """Resolve PGIA140 and bridge older Modbus driver classes that lack id=."""
    module_name = os.environ.get(
        "XINDUN_PGIA_MODULE", "robot_con.PGIA140.PGIA140")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "未找到 PGIA140 真机驱动模块 {!r}。当前仓库只含 PGIA140 仿真模型；"
            "请安装 real.py 使用的硬件驱动，或设置 XINDUN_PGIA_MODULE。"
            .format(module_name)) from exc
    gripper_class = getattr(module, "PGIA140", None)
    if gripper_class is None:
        raise RuntimeError(
            "PGIA140 驱动模块 {!r} 中没有 PGIA140 类。".format(module_name))

    modbus_module = getattr(module, "dh_modbus_gripper", None)
    modbus_class = getattr(modbus_module, "dh_modbus_gripper", None)
    if modbus_class is None:
        raise RuntimeError(
            "PGIA140 驱动没有暴露底层 dh_modbus_gripper 类。")
    try:
        inspect.signature(modbus_class).bind(id=GRIPPER_ID)
    except TypeError:
        legacy_factory = modbus_class

        def make_modbus_gripper(id=1):
            device = legacy_factory()
            device.gripper_ID = int(id)
            return device

        modbus_module.dh_modbus_gripper = make_modbus_gripper

    device_module = getattr(modbus_module, "dh_device", None)
    serial_port = getattr(device_module, "serialPort", None)
    if serial_port is None:
        raise RuntimeError("PGIA140 驱动没有暴露串口对象，无法设置读取超时。")
    serial_port.timeout = GRIPPER_READ_TIMEOUT

    if not getattr(modbus_class, "_xindun_checked_io", False):
        original_write = modbus_class.WriteRegisterFunc
        original_read = modbus_class.ReadRegisterFunc

        def checked_write(self, register, value):
            if not original_write(self, register, value):
                raise TimeoutError(
                    "PGIA140 Modbus 写寄存器 0x{:04X} 无响应；请检查串口、夹爪 ID 和接线。"
                    .format(register))
            return True

        def checked_read(self, register):
            try:
                value = original_read(self, register)
            except UnboundLocalError as exc:
                raise TimeoutError(
                    "PGIA140 Modbus 读寄存器 0x{:04X} 未收到完整响应；"
                    "请检查串口、夹爪 ID 和接线。".format(register)) from exc
            if not isinstance(value, (int, np.integer)):
                raise TimeoutError(
                    "PGIA140 Modbus 读寄存器 0x{:04X} 返回无效响应。".format(register))
            return int(value)

        modbus_class.WriteRegisterFunc = checked_write
        modbus_class.ReadRegisterFunc = checked_read
        modbus_class._xindun_checked_io = True
    return gripper_class


def close_gripper_serial():
    module_name = os.environ.get(
        "XINDUN_PGIA_MODULE", "robot_con.PGIA140.PGIA140")
    module = importlib.import_module(module_name)
    modbus_module = module.dh_modbus_gripper
    serial_port = modbus_module.dh_device.serialPort
    is_open = getattr(serial_port, "is_open", None)
    if is_open is None:
        is_open = serial_port.isOpen()
    if is_open:
        serial_port.close()


def print_plan(plan, start_joints):
    print("FoundationPose 位姿文件: {}".format(plan["pose_file"]))
    print("左臂起始关节角 (rad): {}".format(np.asarray(start_joints).tolist()))
    print("高位等待点 (m): {}".format(plan["hover_pos"].tolist()))
    print("预抓取点 (m): {}".format(plan["pregrasp_pos"].tolist()))
    print("抓取点 (m): {}".format(plan["grasp_pos"].tolist()))
    print("右臂让位目标 TCP (世界坐标 m): {}".format(
        plan["right_clearance"]["target"][:3, 3].tolist()))
    print("关节路径点数: hover={}, pregrasp={}; 下探={} 个 IK 点".format(
        len(plan["hover_path"]), len(plan["pregrasp_path"]),
        len(plan["descent"]) - 1))
    print("关节速度 vel={}, 加速度 acc={}".format(MOVE_VEL, MOVE_ACC))


def execute_grasp(plan, arm, gripper, right_arm):
    from workcell_followup import (build_grasp_approach_path,
                                   execute_parallel_actions, planned_samples)
    if right_arm is None:
        raise ValueError("右臂让位需要已连接的右机械臂。")
    print("阶段 1/3：左臂移动到高位点，同时右臂向世界 -Y 让位。")
    left_path = np.vstack((arm.get_jnt_values(), plan["hover_path"][1:]))
    
    right_path = np.asarray([right_arm.get_jnt_values(), plan["right_wait"]])
    actions = [
        dict(step="hover", arm="lft_arm", path=left_path,
             samples=planned_samples(plan, "left", left_path, MOVE_VEL, MOVE_ACC),
             label="左臂到高位点", event="open"),
        dict(step="clearance", arm="rgt_arm", path=right_path,
             samples=planned_samples(plan, "right", right_path, MOVE_VEL, MOVE_ACC),
             label="右臂向世界 -Y 让位并朝下", event=None),
    ]
    execute_parallel_actions(actions, {"lft_arm": arm, "rgt_arm": right_arm},
                             plan["target"], gripper, MOVE_VEL, MOVE_ACC)

    print("阶段 2/3：连续移动到预抓取点并下探至目标。")
    approach_and_descent = build_grasp_approach_path(plan, arm.get_jnt_values())
    arm.move_jntspace_path(
        approach_and_descent, vel=MOVE_VEL, acc=MOVE_ACC,
        execution_mode="buffer")

    print("闭合左夹爪，完成夹持。")
    gripper.close_g()

    print("保持夹持，接续抬升并前往开瓶器。")


def main():
    parser = argparse.ArgumentParser(
        description="现场采集 D435 位姿，规划抓取、开瓶和移液机械臂动作。")
    parser.add_argument(
        "--execute", action="store_true",
        help="连接真实双臂执行完整流程；右手需预先持有移液枪。")
    args = parser.parse_args()

    arm = None
    right_arm = None
    gripper = None
    try:
        from workcell_followup import (RIGHT_START, build_followup_plan,
                                       execute_followup)
        if args.execute:
            from AUBO.aubo_robot import AUBO

            print("正在连接左右机械臂，确认控制器可用。", flush=True)
            arm = AUBO(robot_ip=LEFT_ROBOT_IP, robot_port=ROBOT_PORT)
            right_arm = AUBO(robot_ip=RIGHT_ROBOT_IP, robot_port=ROBOT_PORT)
            arm.get_jnt_values()
            right_arm.get_jnt_values()

            gripper_class = get_real_gripper_class()
            print("正在初始化左夹爪；等待 PGIA140 Modbus 响应。", flush=True)
            try:
                gripper = gripper_class(
                    port=os.environ.get("XINDUN_GRIPPER_PORT", GRIPPER_PORT),
                    baudrate=GRIPPER_BAUDRATE,
                    force=30,
                    speed=100,
                    id=GRIPPER_ID,
                )
            except Exception:
                close_gripper_serial()
                raise
        else:
            start_joints = CONFIGURED_START.copy()
            right_start = RIGHT_START.copy()

        pose, pose_file = capture_live_pose()
        if args.execute:
            # Use fresh joint readings after camera selection for planning.
            start_joints = arm.get_jnt_values()
            right_start = right_arm.get_jnt_values()
        planning_start = perf_counter()
        with _selective_collision_policy():
            plan = build_grasp_plan(start_joints, pose, pose_file, right_start)
            build_followup_plan(
                plan, MOVE_VEL, MOVE_ACC,
                opener_target_world_x_offset=0.075,
                validate_grasp_transfer=True,
                optimize_merged_paths=True)
        print("规划耗时：{:.2f} 秒。".format(
            perf_counter() - planning_start), flush=True)
        print_plan(plan, start_joints)
        if not args.execute:
            print("仅完成离线规划，未连接或驱动真机。执行时添加 --execute。")
            return

        print("请确认工作区清空、右手已正确夹持移液枪、目标 pose 与相机标定一致、急停可用。")

        # if input("确认以上条件并开始真机动作？输入 EXECUTE 继续：").strip() != "EXECUTE":
        #     print("已取消；未发送运动指令。")
        #     return

        execute_grasp(plan, arm, gripper, right_arm)
        execute_followup(plan, arm, right_arm, gripper, MOVE_VEL, MOVE_ACC)
        print("完整流程完成：培养瓶已放回，左右臂已回到固定初始关节角。")
    finally:
        try:
            if gripper is not None:
                close_gripper_serial()
        finally:
            try:
                if right_arm is not None:
                    right_arm.disconnect()
            finally:
                if arm is not None:
                    arm.disconnect()

if __name__ == "__main__":
    main()
