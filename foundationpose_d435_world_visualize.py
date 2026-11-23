"""Select an object from the live D435 stream, estimate its pose, and show it in WRS."""

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime


FOUNDATIONPOSE_ROOT = Path(r"D:\FoundationPose-main\FoundationPose-main")
FOUNDATIONPOSE_PYTHON = Path(r"D:\conda-envs\foundationpose-win\python.exe")
WRS_ROOT = Path(r"D:\wrs-fujikoshi-main\wrs-fujikoshi-main")
WRS_PYTHON = Path(r"C:\Users\asus\miniconda3\envs\wrs\python.exe")
SAM_SOURCE_ROOT = Path(r"D:\sam_seg\segment-anything-main\segment-anything-main")
SAM_CHECKPOINT = Path(r"D:\sam_seg\SAM-Tool-main\sam_vit_h_4b8939.pth")
DEFAULT_MESH_FILE = WRS_ROOT / "xindun_1110" / "objects" / "cf.STL"
RESULT_ROOT = Path(__file__).resolve().parent / "foundationpose_world_result"
DEFAULT_REFERENCE_ORIENTATION_POSE = (
    RESULT_ROOT / "20261113_220757_649151" / "pose.json"
)

# Raw calibration matrix: calibration camera coordinates -> WRS world/base.
T_WORLD_CAMERA_CALIBRATION = [
    [0.034895, 0.874353, -0.484035, 0.210000],
    [-0.999268, 0.022913, -0.030649, 0.020000],
    [-0.015707, 0.484750, 0.874512, 0.640000],
    [0.000000, 0.000000, 0.000000, 1.000000],
]

# FoundationPose uses the OpenCV optical frame (x right, y down, z forward).
# The supplied calibration appears to use the OpenGL camera frame
# (x right, y up, z backward); this basis change is its own inverse.
T_CALIBRATION_CAMERA_FROM_OPENCV_CAMERA = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

GRIPPER_MAX_OPENING = 0.12
GRASP_JAW_WIDTH = 0.09
INITIAL_ARM_LIFT_HEIGHT = 0.05
PREGRASP_DISTANCE = 0.08
PREGRASP_HEIGHT = 0.20
LIFT_HEIGHT = 0.15
PATH_STEP = 0.015


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a known object from the live D435 stream, estimate its pose, and show it with the I5 dual robot."
    )
    parser.add_argument("--mesh-file", type=Path, default=DEFAULT_MESH_FILE)
    parser.add_argument("--sam-checkpoint", type=Path, default=SAM_CHECKPOINT)
    parser.add_argument("--reference-orientation-pose", type=Path,
                        default=DEFAULT_REFERENCE_ORIENTATION_POSE)
    parser.add_argument("--estimate-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--visualize-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pose-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def run_pipeline(mesh_file, sam_checkpoint):
    mesh_file = mesh_file.resolve()
    if not mesh_file.is_file():
        raise FileNotFoundError("目标网格不存在: {}".format(mesh_file))
    if not SAM_SOURCE_ROOT.is_dir():
        raise FileNotFoundError("SAM 源码目录不存在: {}".format(SAM_SOURCE_ROOT))
    if not sam_checkpoint.is_file():
        raise FileNotFoundError("SAM 权重文件不存在: {}".format(sam_checkpoint))
    if not FOUNDATIONPOSE_PYTHON.is_file():
        raise FileNotFoundError("FoundationPose Python 环境不存在: {}".format(FOUNDATIONPOSE_PYTHON))
    if not WRS_PYTHON.is_file():
        raise FileNotFoundError("WRS Python 环境不存在: {}".format(WRS_PYTHON))

    run_dir = RESULT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=True)
    pose_file = run_dir / "pose.json"
    script_file = Path(__file__).resolve()

    print("启动 FoundationPose + D435 实时视频流；请在视频窗口中标框后确认。")
    subprocess.run(
        [
            str(FOUNDATIONPOSE_PYTHON), str(script_file),
            "--estimate-only", "--mesh-file", str(mesh_file),
            "--sam-checkpoint", str(sam_checkpoint.resolve()),
            "--reference-orientation-pose",
            str(DEFAULT_REFERENCE_ORIENTATION_POSE),
            "--pose-file", str(pose_file), "--output-dir", str(run_dir),
        ],
        cwd=str(FOUNDATIONPOSE_ROOT),
        check=True,
    )

    print("启动 WRS 场景，显示估计物体和 I5 双臂机器人。")
    subprocess.run(
        [
            str(WRS_PYTHON), str(script_file),
            "--visualize-only", "--mesh-file", str(mesh_file),
            "--pose-file", str(pose_file),
        ],
        cwd=str(WRS_ROOT),
        check=True,
    )


def predict_box_mask_with_sam(color_bgr, checkpoint, cv2, np, torch, box):
    """Predict one SAM mask from the box confirmed in the live stream."""
    if not SAM_SOURCE_ROOT.is_dir():
        raise FileNotFoundError("SAM source directory does not exist: {}".format(
            SAM_SOURCE_ROOT
        ))
    if not checkpoint.is_file():
        raise FileNotFoundError("SAM checkpoint does not exist: {}".format(checkpoint))

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SAM_SOURCE_ROOT))
    from segment_anything import SamPredictor, sam_model_registry

    sam = None
    predictor = None
    try:
        sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint.resolve()))
        sam.to(device="cuda")
        predictor = SamPredictor(sam)
        predictor.set_image(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.asarray(box, dtype=np.float32),
            multimask_output=False,
        )
        return masks[int(np.argmax(scores))].astype(bool)
    finally:
        if predictor is not None:
            del predictor
        if sam is not None:
            del sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def select_frame_from_stream(camera, cv2, np, mask_predictor=None):
    """Show live color frames until the user confirms a rectangle and mask."""
    window_name = "D435 live target selection"
    drag_start = [None]
    drag_end = [None]
    selected_box = [None]
    latest_color = [None]
    latest_depth = [None]
    selected_mask = [None]
    selection_frame = [None]
    box_changed = [False]
    selection_frozen = [False]

    def on_mouse(event, x, y, flags, param):
        if selection_frozen[0]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drag_start[0] = (x, y)
            drag_end[0] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drag_start[0] is not None:
            drag_end[0] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and drag_start[0] is not None:
            x0, y0 = drag_start[0]
            drag_start[0] = None
            drag_end[0] = None
            if abs(x - x0) >= 8 and abs(y - y0) >= 8:
                selected_box[0] = np.asarray(
                    [min(x0, x), min(y0, y), max(x0, x), max(y0, y)],
                    dtype=np.float32,
                )
                box_changed[0] = True

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print("D435 实时视频流：左键拖拽框选目标；C/回车锁定当前帧；R 重画；Q 取消。")
    try:
        while True:
            if not selection_frozen[0]:
                color_bgr, depth_m = camera.get_frames()
                if color_bgr is not None and depth_m is not None:
                    latest_color[0] = color_bgr
                    latest_depth[0] = depth_m

            if latest_color[0] is None:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    return None
                continue

            if box_changed[0]:
                box_changed[0] = False
                selection_frame[0] = (
                    latest_color[0].copy(), latest_depth[0].copy()
                )
                if mask_predictor is not None:
                    print("Generating the SAM mask for the selected frame...")
                    selected_mask[0] = mask_predictor(
                        selection_frame[0][0], selected_box[0].copy()
                    )
                latest_color[0], latest_depth[0] = selection_frame[0]
                selection_frozen[0] = True

            preview = latest_color[0].copy()
            if selected_mask[0] is not None:
                overlay = np.zeros_like(preview)
                overlay[selected_mask[0]] = (0, 190, 0)
                preview = cv2.addWeighted(preview, 0.68, overlay, 0.32, 0)
            if selected_box[0] is not None:
                x0, y0, x1, y1 = selected_box[0].astype(int)
                cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 220, 255), 2)
            if drag_start[0] is not None and drag_end[0] is not None:
                cv2.rectangle(preview, drag_start[0], drag_end[0], (255, 180, 0), 2)
            instruction = (
                "Green overlay: SAM mask | C confirm | R resume live and redraw | Q cancel"
                if selection_frozen[0] else
                "Drag a box around target | R redraw | Q cancel"
            )
            cv2.putText(
                preview, instruction,
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 1, cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("r"):
                selected_box[0] = None
                selected_mask[0] = None
                selection_frame[0] = None
                box_changed[0] = False
                selection_frozen[0] = False
            elif key in (ord("c"), 10, 13):
                if selected_box[0] is None:
                    print("请先拖拽一个包含目标的矩形框。")
                    continue
                locked_color = (selection_frame[0][0].copy()
                                if selection_frame[0] is not None
                                else latest_color[0].copy())
                locked_depth = (selection_frame[0][1].copy()
                                if selection_frame[0] is not None
                                else latest_depth[0].copy())
                locked_box = selected_box[0].copy()
                return (
                    locked_color,
                    locked_depth,
                    np.asarray(camera.K, dtype=np.float64).copy(),
                    locked_box,
                    selected_mask[0],
                )
    finally:
        cv2.destroyWindow(window_name)


def select_mask_with_sam(color_bgr, checkpoint, cv2, np, torch, initial_box=None):
    if not SAM_SOURCE_ROOT.is_dir():
        raise FileNotFoundError("SAM 源码目录不存在: {}".format(SAM_SOURCE_ROOT))
    if not checkpoint.is_file():
        raise FileNotFoundError("SAM 权重文件不存在: {}".format(checkpoint))

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SAM_SOURCE_ROOT))
    from segment_anything import SamPredictor, sam_model_registry

    sam = None
    predictor = None
    window_created = False
    try:
        sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint.resolve()))
        sam.to(device="cuda")
        predictor = SamPredictor(sam)
        predictor.set_image(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB))

        window_name = "SAM target mask"
        prompts = []
        labels = []
        prompt_changed = [initial_box is not None]
        selected_mask = None
        selected_score = 0.0
        selected_box = [(
            np.asarray(initial_box, dtype=np.float32).copy()
            if initial_box is not None else None
        )]
        drag_start = [None]
        drag_end = [None]

        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                drag_start[0] = (x, y)
                drag_end[0] = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and drag_start[0] is not None:
                drag_end[0] = (x, y)
            elif event == cv2.EVENT_RBUTTONDOWN:
                prompts.append((x, y))
                labels.append(0)
                prompt_changed[0] = True
            elif event == cv2.EVENT_LBUTTONUP and drag_start[0] is not None:
                x0, y0 = drag_start[0]
                drag_start[0] = None
                drag_end[0] = None
                if abs(x - x0) >= 8 and abs(y - y0) >= 8:
                    selected_box[0] = np.asarray(
                        [min(x0, x), min(y0, y), max(x0, x), max(y0, y)],
                        dtype=np.float32,
                    )
                    prompt_changed[0] = True
                else:
                    prompts.append((x, y))
                    labels.append(1)
                    prompt_changed[0] = True

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        window_created = True
        cv2.setMouseCallback(window_name, on_mouse)
        print("SAM 掩码确认：可继续添加正负点或重画框；C 确认，U 撤销，R 重置，Q 取消。")

        while True:
            if prompt_changed[0]:
                prompt_changed[0] = False
                selected_mask = None
                if 1 in labels or selected_box[0] is not None:
                    point_coords = (
                        np.asarray(prompts, dtype=np.float32) if prompts else None
                    )
                    point_labels = (
                        np.asarray(labels, dtype=np.int32) if labels else None
                    )
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        box=selected_box[0],
                        multimask_output=(len(prompts) == 1),
                    )
                    best_index = int(np.argmax(scores))
                    selected_mask = masks[best_index].astype(bool)
                    selected_score = float(scores[best_index])

            preview = color_bgr.copy()
            if selected_mask is not None:
                overlay = np.zeros_like(preview)
                overlay[selected_mask] = (0, 190, 0)
                preview = cv2.addWeighted(preview, 0.68, overlay, 0.32, 0)
            for (x, y), label in zip(prompts, labels):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(preview, (x, y), 5, color, -1)
            if selected_box[0] is not None:
                x0, y0, x1, y1 = selected_box[0].astype(int)
                cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 180, 0), 2)
            if drag_start[0] is not None and drag_end[0] is not None:
                cv2.rectangle(preview, drag_start[0], drag_end[0], (255, 180, 0), 2)
            prompt_status = "Left click: object  Left drag: box  Right click: background"
            key_status = "U: undo  R: reset  C: confirm  Q: cancel"
            cv2.putText(preview, prompt_status, (8, preview.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(preview, key_status, (8, preview.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1, cv2.LINE_AA)
            if selected_mask is not None:
                cv2.putText(preview, "SAM score: {:.3f}".format(selected_score),
                            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return None
            if key == ord("u"):
                if selected_box[0] is not None:
                    selected_box[0] = None
                    prompt_changed[0] = True
                elif prompts:
                    prompts.pop()
                    labels.pop()
                    prompt_changed[0] = True
            elif key == ord("r"):
                prompts.clear()
                labels.clear()
                selected_box[0] = None
                prompt_changed[0] = True
            elif key == ord("c"):
                if selected_mask is None:
                    print("请先添加目标正点或画一个包含目标的矩形框。")
                    continue
                return selected_mask.astype(np.uint8)
    finally:
        if window_created:
            cv2.destroyWindow("SAM target mask")
        del predictor
        del sam
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def validate_t_world_camera(np, transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("T_WORLD_CAMERA 必须是有限的 4x4 矩阵。")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("T_WORLD_CAMERA 最后一行必须为 [0, 0, 0, 1]。")

    rotation = transform[:3, :3]
    orthogonality_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if orthogonality_error > 1e-3 or abs(determinant - 1.0) > 1e-3:
        raise ValueError(
            "T_WORLD_CAMERA 旋转部分不是合法旋转矩阵：det={:.6f}, error={:.6g}".format(
                determinant, orthogonality_error
            )
        )

    optical_z_world = rotation[:, 2]
    print("相机光学 +Z 在 WRS base 中的方向: {}".format(optical_z_world.tolist()))
    if optical_z_world[2] > -0.5:
        print("警告：相机光轴没有明显指向 WRS -Z；请核对相机坐标系与外参。")
    return transform


def interpolate_joint_path(start, goal, sample_count, np):
    delta = (goal - start + np.pi) % (2.0 * np.pi) - np.pi
    return [start + delta * (index / sample_count)
            for index in range(1, sample_count + 1)]


def plan_cartesian_path(robot, obj, start_pos, goal_pos, start_conf,
                        rotation, np, check_object=True):
    distance = float(np.linalg.norm(goal_pos - start_pos))
    sample_count = max(1, int(np.ceil(distance / PATH_STEP)))
    configurations = []
    previous = start_conf.copy()
    for index in range(1, sample_count + 1):
        target_pos = start_pos + (goal_pos - start_pos) * (index / sample_count)
        solution = robot.tracik(
            target_pos, rotation, "lft_arm", seed_jnt_values=previous
        )
        if solution is None:
            return None
        robot.fk("lft_arm", solution)
        if robot.is_collided():
            return None
        if check_object and robot.is_object_collided(obj, contact_hand="lft_arm"):
            return None
        configurations.append(solution)
        previous = solution
    return configurations


def plan_left_grasp(robot, obj, pose_world_object, np):
    rotation = pose_world_object[:3, :3]
    position = pose_world_object[:3, 3]
    grasp_pos = position + rotation @ np.array([0.0, 0.0, 0.018])
    approach_dir = -rotation[:, 2]
    pregrasp_pos = grasp_pos - approach_dir * PREGRASP_DISTANCE
    hover_pos = grasp_pos + np.array([0.0, 0.0, PREGRASP_HEIGHT])
    lift_pos = grasp_pos + np.array([0.0, 0.0, LIFT_HEIGHT])
    opening_axis = rotation[:, 1]
    tool_rotation = np.column_stack((
        np.cross(opening_axis, approach_dir), opening_axis, approach_dir
    ))

    # Match the updated gripper and the CAD body's 90 mm jaw span.
    robot.lft_hnd.jawwidth_rng = [0.0, GRIPPER_MAX_OPENING]
    robot.lft_hnd.lft.jnts[1]["motion_rng"] = np.array(
        [0.0, GRIPPER_MAX_OPENING / 2.0]
    )
    robot.lft_hnd.rgt.jnts[1]["motion_rng"] = np.array(
        [0.0, GRIPPER_MAX_OPENING / 2.0]
    )

    start_conf = robot.get_jnt_values("lft_arm").copy()
    initial_tcp_pos, initial_tcp_rot = robot.get_gl_tcp("lft_arm")
    initial_lift_pos = initial_tcp_pos + np.array(
        [0.0, 0.0, INITIAL_ARM_LIFT_HEIGHT]
    )
    robot.jaw_to("lft_hnd", GRIPPER_MAX_OPENING)
    initial_lift = plan_cartesian_path(
        robot, obj, initial_tcp_pos, initial_lift_pos, start_conf,
        initial_tcp_rot, np,
    )
    if initial_lift is None:
        robot.fk("lft_arm", start_conf)
        raise RuntimeError("左臂初始上抬 {:.0f} mm 的直线路径发生碰撞或无法到达。".format(
            INITIAL_ARM_LIFT_HEIGHT * 1000.0
        ))
    lifted_conf = initial_lift[-1] if initial_lift else start_conf

    rng = np.random.default_rng(20261112)
    seeds = [lifted_conf] + [rng.uniform(-np.pi, np.pi, 6) for _ in range(48)]

    for seed_index, seed in enumerate(seeds):
        robot.fk("lft_arm", lifted_conf)
        hover_conf = robot.tracik(
            hover_pos, tool_rotation, "lft_arm", seed_jnt_values=seed
        )
        if hover_conf is None:
            continue
        hover_conf = np.asarray(hover_conf, dtype=float).copy()
        hover_conf += 2 * np.pi * np.round((lifted_conf - hover_conf) / (2 * np.pi))
        if (not np.isfinite(hover_conf).all()
                or not robot.is_jnt_values_in_ranges("lft_arm", hover_conf)):
            continue

        transition = interpolate_joint_path(lifted_conf, hover_conf, 30, np)
        robot.jaw_to("lft_hnd", GRIPPER_MAX_OPENING)
        transition_clear = True
        for conf in transition:
            robot.fk("lft_arm", conf)
            if (robot.is_collided()
                    or robot.is_object_collided(obj, contact_hand="lft_arm")):
                transition_clear = False
                break
        if not transition_clear:
            continue

        hover_to_pregrasp = plan_cartesian_path(
            robot, obj, hover_pos, pregrasp_pos, hover_conf,
            tool_rotation, np,
        )
        if hover_to_pregrasp is None:
            continue
        pregrasp_conf = hover_to_pregrasp[-1] if hover_to_pregrasp else hover_conf
        pregrasp_to_grasp = plan_cartesian_path(
            robot, obj, pregrasp_pos, grasp_pos, pregrasp_conf,
            tool_rotation, np,
        )
        if pregrasp_to_grasp is None:
            continue

        grasp_conf = pregrasp_to_grasp[-1]
        robot.fk("lft_arm", grasp_conf)
        robot.jaw_to("lft_hnd", GRASP_JAW_WIDTH)
        robot.hold("lft_hnd", obj, jawwidth=GRASP_JAW_WIDTH)
        grasp_to_lift = plan_cartesian_path(
            robot, obj, grasp_pos, lift_pos, grasp_conf,
            tool_rotation, np, check_object=False,
        )
        if grasp_to_lift is None:
            robot.release("lft_hnd", obj, jawwidth=GRIPPER_MAX_OPENING)
            continue

        plan = {
            "seed_index": seed_index,
            "initial_lift": initial_lift,
            "transition": transition,
            "hover_to_pregrasp": hover_to_pregrasp,
            "pregrasp_to_grasp": pregrasp_to_grasp,
            "grasp_to_lift": grasp_to_lift,
            "grasp_pos": grasp_pos,
            "hover_pos": hover_pos,
            "lift_pos": lift_pos,
            "rotation": tool_rotation,
        }
        robot.release("lft_hnd", obj, jawwidth=GRIPPER_MAX_OPENING)
        obj.set_pose(position, rotation)
        robot.fk("lft_arm", start_conf)
        return plan

    obj.set_pose(position, rotation)
    robot.fk("lft_arm", start_conf)
    raise RuntimeError(
        "左臂未找到无碰撞的抓取路径；请检查目标位姿和左右臂初始姿态。"
    )


def attach_live_robot_visuals(base, robot):
    chains = (
        robot.base_stand, robot.base_stand2, robot.base_stand3,
        robot.lft_arm.jlc, robot.rgt_arm.jlc,
        robot.lft_hnd.coupling, robot.lft_hnd.lft, robot.lft_hnd.rgt,
        robot.rgt_hnd.coupling, robot.rgt_hnd.lft, robot.rgt_hnd.rgt,
    )
    bindings = []
    for chain in chains:
        for link in chain.lnks:
            model = link.get("collision_model")
            if model is None:
                continue
            visual = model.copy()
            visual.attach_to(base)
            bindings.append((link, visual))

    def update():
        for link, visual in bindings:
            visual.set_pose(link["gl_pos"], link["gl_rotmat"])

    update()
    return update


def animate_left_grasp(base, robot, obj, plan, update_visuals,
                       object_frame, data_adapter):
    from direct.task import Task

    path = (plan["initial_lift"] + plan["transition"]
            + plan["hover_to_pregrasp"]
            + plan["pregrasp_to_grasp"] + plan["grasp_to_lift"])
    grasp_frame = (len(plan["initial_lift"])
                   + len(plan["transition"])
                   + len(plan["hover_to_pregrasp"])
                   + len(plan["pregrasp_to_grasp"]) - 1)
    index = [0]
    held = [False]

    def update(task):
        if index[0] >= len(path):
            print("抓取后物体坐标系原点 (WRS base/world, m): {}".format(
                obj.get_pos().tolist()
            ))
            return Task.done
        robot.fk("lft_arm", path[index[0]])
        if index[0] == grasp_frame:
            robot.jaw_to("lft_hnd", GRASP_JAW_WIDTH)
            robot.hold("lft_hnd", obj, jawwidth=GRASP_JAW_WIDTH)
            held[0] = True
        elif not held[0]:
            robot.jaw_to("lft_hnd", GRIPPER_MAX_OPENING)
        robot.get_oih_list("lft_hnd")
        frame_pos = obj.get_pos()
        object_frame.objpdnp.setPos(*frame_pos.tolist())
        object_frame.objpdnp.setQuat(
            data_adapter.npmat3_to_pdquat(obj.get_rotmat())
        )
        update_visuals()
        index[0] += 1
        return Task.cont

    base.taskMgr.doMethodLater(0.04, update, "left-arm-cf-grasp")


def estimate_once(mesh_file, pose_file, output_dir, sam_checkpoint,
                  reference_orientation_pose):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(FOUNDATIONPOSE_ROOT))

    import cv2
    import numpy as np
    import torch
    import trimesh
    import nvdiffrast.torch as dr
    from D435test import RealSenseD435
    from estimater import FoundationPose
    from learning.training.predict_pose_refine import PoseRefinePredictor
    from learning.training.predict_score import ScorePredictor

    if not torch.cuda.is_available():
        raise RuntimeError("FoundationPose 需要可用的 CUDA GPU。")

    mesh_file = mesh_file.resolve()
    pose_file = pose_file.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_orientation_pose = reference_orientation_pose.resolve()
    with reference_orientation_pose.open("r", encoding="utf-8") as stream:
        reference_payload = json.load(stream)
    reference_pose_world = np.asarray(
        reference_payload.get("T_world_object"), dtype=np.float64
    )
    if (reference_pose_world.shape != (4, 4)
            or not np.isfinite(reference_pose_world).all()):
        raise ValueError("参考姿态的 T_world_object 必须是有限的 4x4 矩阵。")
    reference_z_world = reference_pose_world[:3, 2]
    reference_z_world /= np.linalg.norm(reference_z_world)

    t_world_camera_calibration = np.asarray(
        T_WORLD_CAMERA_CALIBRATION, dtype=np.float64
    )
    t_calibration_camera_from_opencv_camera = np.asarray(
        T_CALIBRATION_CAMERA_FROM_OPENCV_CAMERA, dtype=np.float64
    )
    t_world_camera = validate_t_world_camera(
        np,
        t_world_camera_calibration @ t_calibration_camera_from_opencv_camera,
    )
    expected_z_camera = t_world_camera[:3, :3].T @ reference_z_world
    expected_z_camera /= np.linalg.norm(expected_z_camera)
    z_axis_tolerance_deg = 15.0

    mesh = trimesh.load(str(mesh_file), force="mesh", process=False)
    extents = np.asarray(mesh.extents, dtype=np.float64)
    print("目标网格: {}".format(mesh_file))
    print("网格尺寸 (m): {}".format(np.array2string(extents, precision=4)))
    if float(extents.max()) > 2.0:
        raise ValueError("网格看起来不是米制尺寸；FoundationPose 网格必须与深度统一使用米。")

    camera = None
    try:
        print("启动 D435 实时视频流，等待你把目标放好并标框。")
        camera = RealSenseD435(width=640, height=480, fps=30)
        selected = select_frame_from_stream(
            camera,
            cv2,
            np,
            mask_predictor=lambda color, box: predict_box_mask_with_sam(
                color, sam_checkpoint, cv2, np, torch, box
            ),
        )
        if selected is None:
            raise RuntimeError("用户取消了实时选框。")
        color_bgr, depth_m, camera_K, selected_box, sam_mask = selected
        camera.stop()
        camera = None

        if sam_mask is None:
            raise RuntimeError("没有确认 SAM 目标掩码，已取消估计。")
        sam_pixels = int(sam_mask.sum())
        if sam_pixels < 500:
            raise RuntimeError("SAM 目标掩码少于 500 个像素；请重新选择。")

        valid_depth = np.isfinite(depth_m) & (depth_m > 0)
        mask = (sam_mask.astype(bool) & valid_depth).astype(np.uint8)
        valid_pixels = int(mask.sum())
        depth_support = valid_pixels / float(sam_pixels)
        print("SAM 掩码像素: {}, 其中有效深度: {} ({:.1%})".format(
            sam_pixels, valid_pixels, depth_support
        ))
        if valid_pixels < 500:
            raise RuntimeError(
                "SAM 掩码内有效深度少于 500 像素；检查目标反光、距离或重新选择。"
            )

        cv2.imwrite(str(output_dir / "sam_mask.png"), sam_mask * 255)
        cv2.imwrite(str(output_dir / "mask.png"), mask * 255)
        cv2.imwrite(str(output_dir / "color.png"), color_bgr)
        depth_mm = np.clip(np.rint(depth_m * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(output_dir / "depth_mm.png"), depth_mm)

        cv2.destroyAllWindows()

        print("释放 SAM 显存后加载 FoundationPose。")
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        estimator = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=str(output_dir / "debug"),
            debug=0,
            glctx=glctx,
            diameter_n_sample=2000,
            rotation_grid_min_n_views=20,
            rotation_grid_inplane_step=90,
            rotation_grid_max_n_views=20,
            registration_z_axis_camera=expected_z_camera,
            registration_z_axis_tolerance_deg=z_axis_tolerance_deg,
        )

        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        print("运行 FoundationPose 首帧注册，并按参考样本筛选物体 Z 轴方向。")
        estimator.register(
            K=camera_K,
            rgb=color_rgb,
            depth=depth_m,
            ob_mask=mask,
            iteration=2,
        )
        candidate_poses = estimator.poses.detach().cpu().numpy()
        candidate_scores = estimator.scores.detach().cpu().numpy().reshape(-1)
        candidate_z_axes = candidate_poses[:, :3, 2]
        candidate_z_norms = np.linalg.norm(candidate_z_axes, axis=1)
        valid_z = np.isfinite(candidate_z_axes).all(axis=1) & (candidate_z_norms > 1e-9)
        candidate_dots = np.full(len(candidate_poses), -1.0, dtype=np.float64)
        candidate_dots[valid_z] = (
            candidate_z_axes[valid_z] / candidate_z_norms[valid_z, None]
        ) @ expected_z_camera
        eligible = np.flatnonzero(
            candidate_dots >= np.cos(np.deg2rad(z_axis_tolerance_deg))
        )
        if not len(eligible):
            best_error = float(np.degrees(np.arccos(np.clip(
                candidate_dots.max(), -1.0, 1.0
            ))))
            raise RuntimeError(
                "没有 Z 轴符合参考方向的识别候选；最近候选相差 {:.1f} 度。"
                .format(best_error)
            )
        selected_index = int(eligible[np.argmax(candidate_scores[eligible])])
        selected_candidate_error_deg = float(np.degrees(np.arccos(np.clip(
            candidate_dots[selected_index], -1.0, 1.0
        ))))
        pose_camera_object = (
            estimator.poses[selected_index] @ estimator.get_tf_to_centered_mesh()
        ).detach().cpu().numpy()
        print("旋转候选 {} 个，符合 Z 轴方向 {} 个，选中评分最高候选；偏差 {:.2f} 度。"
              .format(len(candidate_poses), len(eligible), selected_candidate_error_deg))
        if pose_camera_object.shape != (4, 4) or not np.isfinite(pose_camera_object).all():
            raise RuntimeError("FoundationPose 返回了无效的 4x4 位姿。")

        pose_world_object = t_world_camera @ pose_camera_object
        payload = {
            "T_world_camera_calibration": t_world_camera_calibration.tolist(),
            "T_calibration_camera_from_opencv_camera": (
                t_calibration_camera_from_opencv_camera.tolist()
            ),
            "T_world_camera": t_world_camera.tolist(),
            "T_camera_object": pose_camera_object.tolist(),
            "T_world_object": pose_world_object.tolist(),
            "position_world_m": pose_world_object[:3, 3].tolist(),
            "reference_object_z_world": reference_z_world.tolist(),
            "selected_candidate_z_axis_error_deg": selected_candidate_error_deg,
            "z_axis_tolerance_deg": z_axis_tolerance_deg,
        }
        pose_file.parent.mkdir(parents=True, exist_ok=True)
        with pose_file.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)

        print("T_camera_object:\n{}".format(pose_camera_object))
        print("T_world_object:\n{}".format(pose_world_object))
        print("物体在 WRS base/world 中的位置 (m): {}".format(payload["position_world_m"]))
        print("本次彩色图、深度图和掩码已保存到: {}".format(output_dir))
    finally:
        if camera is not None:
            camera.stop()
        cv2.destroyAllWindows()


def visualize(mesh_file, pose_file):
    sys.path.insert(0, str(WRS_ROOT))

    import numpy as np
    import basis.data_adapter as da
    import modeling.collision_model as cm
    import modeling.geometric_model as gm
    import visualization.panda.world as wd
    from robot_sim.robots.i5.i5_dual_427_2 import I5Dual

    mesh_file = mesh_file.resolve()
    with pose_file.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    pose_world_object = np.asarray(payload["T_world_object"], dtype=np.float64)
    t_world_camera = np.asarray(payload["T_world_camera"], dtype=np.float64)

    base = wd.World(cam_pos=[1.55, 1.45, 1.35], lookat_pos=[0.30, 0.0, 0.30])
    gm.gen_frame(pos=[0, 0, 0], rotmat=np.eye(3), length=0.25, thickness=0.006).attach_to(base)

    robot = I5Dual(enable_cc=True)
    robot.fk("lft_arm", np.asarray(
        [1.24405645, -0.5328483, 1.91676323, 3.0209546, 1.17882897, -1.96504525],
        dtype=np.float64,
    ))
    robot.fk("rgt_arm", np.asarray(
        [-0.99680635, 0.59159202, -2.07874146, -2.27002831, -1.71148785, 0.03948051],
        dtype=np.float64,
    ))

    obj = cm.CollisionModel(str(mesh_file), cdprimit_type="box", expand_radius=0.001)
    obj.set_pos(pose_world_object[:3, 3])
    obj.set_rotmat(pose_world_object[:3, :3])
    obj.set_rgba([0.95, 0.55, 0.12, 1.0])
    obj.attach_to(base)
    object_frame = gm.gen_frame(
        pos=np.zeros(3),
        rotmat=np.eye(3),
        length=0.12,
        thickness=0.004,
    )
    object_frame.attach_to(base)
    object_frame.objpdnp.setPos(*pose_world_object[:3, 3].tolist())
    object_frame.objpdnp.setQuat(
        da.npmat3_to_pdquat(pose_world_object[:3, :3])
    )
    gm.gen_frame(
        pos=t_world_camera[:3, 3],
        rotmat=t_world_camera[:3, :3],
        length=0.10,
        thickness=0.003,
    ).attach_to(base)

    print("物体位姿已经变换到 WRS base/world。")
    print("位置 (m): {}".format(pose_world_object[:3, 3].tolist()))

    if mesh_file.name.lower() == "cf.stl":
        plan = plan_left_grasp(robot, obj, pose_world_object, np)
        robot.jaw_to("lft_hnd", GRIPPER_MAX_OPENING)
        update_visuals = attach_live_robot_visuals(base, robot)
        animate_left_grasp(
            base, robot, obj, plan, update_visuals, object_frame, da
        )
        print("左臂抓取路径已规划：先到物体上方，垂直下探，夹紧后抬升 {:.0f} mm。".format(
            LIFT_HEIGHT * 1000.0
        ))
        print("抓取前先竖直上抬 {:.0f} mm。".format(
            INITIAL_ARM_LIFT_HEIGHT * 1000.0
        ))
        print("夹爪规划开度: {:.0f} mm；夹持宽度: {:.0f} mm。".format(
            GRIPPER_MAX_OPENING * 1000.0, GRASP_JAW_WIDTH * 1000.0
        ))
    else:
        robot.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(base)

    print("关闭 Panda3D 窗口退出。")
    base.run()


def main():
    args = parse_args()
    if args.estimate_only:
        if (args.pose_file is None or args.output_dir is None
                or args.reference_orientation_pose is None):
            raise ValueError("内部估计模式缺少 pose-file、output-dir 或参考姿态。")
        estimate_once(args.mesh_file, args.pose_file, args.output_dir,
                      args.sam_checkpoint, args.reference_orientation_pose)
        return
    if args.visualize_only:
        if args.pose_file is None:
            raise ValueError("内部显示模式缺少 pose-file。")
        visualize(args.mesh_file, args.pose_file)
        return
    run_pipeline(args.mesh_file, args.sam_checkpoint)


if __name__ == "__main__":
    main()
