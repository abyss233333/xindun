"""Right-click Run: D435 capture -> complete workcell simulation, without hardware.

Optional --pose-file reuses a raw FoundationPose pose.json for offline review.
--check-only plans and validates without opening an animation window.
"""
import argparse
import math
import os
from time import perf_counter

import numpy as np

import real_grasp as grasp
from workcell_followup import (OPENER_COLLISION_POSITIONS, RIGHT_START,
                               SAMPLE_PERIOD, apply_event, build_followup_plan,
                               build_grasp_approach_path,
                               concatenate_joint_paths, homo,
                               merge_followup_actions, reset_scene,
                               planned_samples, synchronize_samples,
                               timed_samples)

def _fit_samples(samples, length):
    samples = np.asarray(samples, dtype=float)
    if len(samples) >= length:
        return samples[:length].copy()
    return np.concatenate((samples, np.repeat(samples[-1:], length - len(samples), axis=0)))


def animation_actions(plan):
    """Mirror the real command order and continuous grasp path groups."""
    actions = []

    def move(label, arm, path, kind, event=None, samples=None):
        path = np.asarray(path)
        actions.append(dict(label=label, arm=arm, path=path, kind=kind, event=event,
                            samples=(timed_samples(path, grasp.MOVE_VEL, grasp.MOVE_ACC)
                                     if samples is None else
                                     np.asarray(samples, dtype=float).copy())))

    left_high_path = plan["hover_path"]
    right_clearance_path = plan["right_clearance"]["path"][[0, -1]]
    initial_pair = [
        dict(step="hover", arm="lft_arm", path=left_high_path,
             samples=planned_samples(plan, "left", left_high_path,
                                     grasp.MOVE_VEL, grasp.MOVE_ACC),
             label="Move left arm to high waiting pose", event="open"),
        dict(step="clearance", arm="rgt_arm", path=right_clearance_path,
             samples=planned_samples(plan, "right", right_clearance_path,
                                     grasp.MOVE_VEL, grasp.MOVE_ACC),
             label="Right arm: shift world -Y, gripper down", event=None),
    ]
    initial_samples = synchronize_samples([item["samples"] for item in initial_pair])
    actions.append(dict(
        label="Left high pose and right-arm clearance: simultaneous",
        samples=initial_samples[0], parallel_actions=initial_pair,
        parallel_samples=initial_samples, event=None))
    approach_and_descent = build_grasp_approach_path(plan)
    move("Move continuously to pregrasp and descend", "lft_arm",
         approach_and_descent, "linear", "hold")
    followup = merge_followup_actions(plan["followup"], grasp.MOVE_VEL,
                                      grasp.MOVE_ACC)
    index = 0
    if followup and followup[0]["step"] == 5:
        lift_and_step5 = plan.get("lift_and_step5_path")
        if lift_and_step5 is None:
            lift_and_step5 = concatenate_joint_paths(
                np.asarray(plan["descent"])[::-1], followup[0]["path"])
        move("Lift after grasp and continue through reference step 5",
             "lft_arm", lift_and_step5, "path",
             samples=plan.get("lift_and_step5_samples"))
        index = 1
    else:
        move("Lift to pregrasp", "lft_arm", plan["descent"][::-1], "linear")
    while index < len(followup):
        action = followup[index]
        if action.get("parallel_group") == "return_home":
            if (index + 1 >= len(followup)
                    or followup[index + 1].get("parallel_group") != "return_home"
                    or {action["arm"], followup[index + 1]["arm"]}
                    != {"lft_arm", "rgt_arm"}):
                raise RuntimeError("左右臂回位动作未组成有效的同步组。")
            paired = [dict(action), dict(followup[index + 1])]
            parallel_samples = synchronize_samples(
                [item["samples"] for item in paired])
            for item, samples in zip(paired, parallel_samples):
                item["samples"] = samples
            actions.append(dict(
                label="Reference steps 33-34: both arms return home together",
                samples=parallel_samples[0], parallel_actions=paired,
                parallel_samples=parallel_samples, event=None,
            ))
            index += 2
            continue
        if (action["step"] == 8 and index + 1 < len(followup)
                and followup[index + 1]["step"] == 9):
            paired = [dict(action), dict(followup[index + 1])]
            parallel_samples = synchronize_samples(
                [item["samples"] for item in paired]
            )
            actions.append(dict(
                label="Reference steps 8 and 9: both arms move together",
                samples=parallel_samples[0],
                parallel_actions=paired,
                parallel_samples=parallel_samples,
                event=None,
            ))
            index += 2
            continue
        if (tuple(action.get("steps", (action["step"],))) == (20, 21)
                and index + 1 < len(followup)
                and tuple(followup[index + 1].get(
                    "steps", (followup[index + 1]["step"],))) == (22, 23)):
            right_action, left_action = action, followup[index + 1]
            if index + 2 < len(followup) and followup[index + 2]["step"] == 24:
                continuation = followup[index + 2]
                right_samples = np.asarray(right_action["samples"], dtype=float)
                left_samples = np.asarray(left_action["samples"], dtype=float)
                continuation_samples = np.asarray(continuation["samples"],
                                                   dtype=float)
                boundary_index = int(right_action.get("sync_sample_index", -1))
                if boundary_index < 0 or boundary_index >= len(right_samples):
                    sync_conf = np.asarray(right_action["sync_conf"], dtype=float)
                    boundary_index = int(np.argmin(np.max(
                        np.abs(right_samples - sync_conf), axis=1)))
                left_start = np.repeat(left_samples[:1], boundary_index, axis=0)
                left_timeline = np.concatenate((left_start, left_samples,
                                                continuation_samples[1:]), axis=0)
                release_index = boundary_index + len(left_samples) - 1
                timeline_length = max(len(right_samples), len(left_timeline))
                right_timeline = _fit_samples(right_samples, timeline_length)
                left_timeline = _fit_samples(left_timeline, timeline_length)
                paired = [dict(right_action, event=None, samples=right_timeline),
                          dict(left_action, event=None, samples=left_timeline)]
                actions.append(dict(
                    label="Right steps 20-21 continuous; left steps 22-24 start at the step 20 waypoint",
                    samples=right_timeline,
                    parallel_actions=paired,
                    parallel_samples=[right_timeline, left_timeline],
                    indexed_events=[(release_index, "release")],
                    release_index=release_index,
                    continuous_timeline=True,
                    event=None,
                ))
                index += 3
            else:
                paired = [dict(right_action), dict(left_action)]
                parallel_samples = synchronize_samples(
                    [item["samples"] for item in paired]
                )
                actions.append(dict(
                    label="Reference steps 20-21 and 22-23: both arms move together",
                    samples=parallel_samples[0], parallel_actions=paired,
                    parallel_samples=parallel_samples, event=None,
                ))
                index += 2
            continue
        copy = dict(action)
        steps = action.get("steps")
        step_label = (f"steps {steps[0]}-{steps[-1]}"
                      if steps else f"step {action['step']}")
        total_steps = 34 if action["step"] >= 30 else 29
        copy["label"] = f"Reference {step_label} / {total_steps} ({action['arm']})"
        if action["kind"] == "joint":
            copy["samples"] = timed_samples(action["path"][[0, -1]],
                                             grasp.MOVE_VEL, grasp.MOVE_ACC)
        actions.append(copy)
        index += 1
    return actions


class WorkflowAnimation:
    def __init__(self, base, plan, actions, speed=1.):
        from direct.gui.OnscreenText import OnscreenText
        self.base, self.plan, self.actions = base, plan, actions
        self.robot = plan["robot"]
        self.stride = max(1, int(round(.04 * speed / SAMPLE_PERIOD)))
        self._prepare_release_holds()
        self.index, self.point, self.pause = 0, 0, 0
        self.fired_events = set()
        self.paused, self.finished, self.mesh = False, False, None
        self.text = OnscreenText(text="", pos=(-1.28, .91), align=0,
                                 scale=.045, mayChange=True, fg=(0, 0, 0, 1))
        reset_scene(plan)
        self.render()
        base.accept("space", self.toggle)
        base.accept("r", self.restart)
        base.taskMgr.doMethodLater(.04, self.update, "live-workflow-animation")

    def _prepare_release_holds(self):
        """Pause only the left arm briefly for the release while the right advances."""
        for action in self.actions:
            if "release_index" not in action:
                continue
            paired, samples = action["parallel_actions"], action["parallel_samples"]
            if len(paired) != 2:
                continue
            left_index = next(i for i, item in enumerate(paired)
                              if item["arm"] == "lft_arm")
            right_index = next(i for i, item in enumerate(paired)
                               if item["arm"] == "rgt_arm")
            left_samples = np.asarray(samples[left_index], dtype=float)
            right_samples = np.asarray(samples[right_index], dtype=float)
            hold_count = 12 * self.stride
            release_index = action["release_index"]
            left_samples = np.concatenate((left_samples[:release_index + 1],
                np.repeat(left_samples[release_index:release_index + 1], hold_count, axis=0),
                left_samples[release_index + 1:]))
            length = max(len(left_samples), len(right_samples))
            left_samples = _fit_samples(left_samples, length)
            right_samples = _fit_samples(right_samples, length)
            action["parallel_samples"][left_index] = left_samples
            action["parallel_samples"][right_index] = right_samples
            action["parallel_actions"][left_index]["samples"] = left_samples
            action["parallel_actions"][right_index]["samples"] = right_samples
            action["samples"] = right_samples

    def toggle(self):
        self.paused = not self.paused

    def restart(self):
        reset_scene(self.plan)
        self.index, self.point, self.pause = 0, 0, 0
        self.paused, self.finished = False, False
        self.fired_events.clear()
        self.render()

    def render(self):
        if self.mesh is not None:
            self.mesh.detach()
        self.robot.get_oih_list()
        self.mesh = self.robot.gen_meshmodel()
        self.mesh.attach_to(self.base)
        # gen_meshmodel already contains copies of held objects. Only attach
        # stationary originals, otherwise the held meshes are drawn twice.
        held = self.robot.get_oih_list()
        for obj in (self.plan["target"], self.plan["pipette"]):
            if any(obj is item for item in held):
                obj.detach()
            else:
                obj.attach_to(self.base)

    def update(self, task):
        if self.paused or self.finished:
            return task.again
        if self.pause:
            self.pause -= 1
            return task.again
        if self.index >= len(self.actions):
            self.text.setText("Complete: both arms returned home\nR: replay | Space: pause")
            self.finished = True
            return task.again
        action = self.actions[self.index]
        path = action["samples"]
        previous_point = self.point
        self.point = min(self.point + self.stride, len(path) - 1)
        parallel_actions = action.get("parallel_actions")
        if parallel_actions is None:
            self.robot.fk(action["arm"], path[self.point])
        else:
            for paired_action, paired_samples in zip(
                    parallel_actions, action["parallel_samples"]):
                self.robot.fk(paired_action["arm"], paired_samples[self.point])
        self.robot.get_oih_list()
        for event_index, event in action.get("indexed_events", []):
            event_key = (self.index, event_index)
            if (previous_point < event_index <= self.point
                    and event_key not in self.fired_events):
                apply_event(self.robot, self.plan["target"], event)
                self.fired_events.add(event_key)
        if self.point == len(path) - 1:
            if parallel_actions is not None:
                for paired_action in parallel_actions:
                    if paired_action["event"] == "open":
                        self.robot.jaw_to("lft_arm", grasp.LFT_OPEN_WIDTH)
                    else:
                        apply_event(self.robot, self.plan["target"],
                                    paired_action["event"])
            elif action["event"] == "open":
                self.robot.jaw_to("lft_arm", grasp.LFT_OPEN_WIDTH)
            else:
                apply_event(self.robot, self.plan["target"], action["event"])
            self.index += 1
            self.point = 0
            has_event = (any(item["event"] for item in parallel_actions)
                         if parallel_actions is not None else bool(action["event"]))
            self.pause = (0 if action.get("continuous_timeline")
                          else 12 if has_event else 3)
        self.text.setText(action["label"] + "\nSpace: pause | R: replay")
        self.render()
        return task.again


def add_station_models(base, plan):
    plan["table_obstacle"].attach_to(base)
    for position in OPENER_COLLISION_POSITIONS:
        opener = grasp.cm.CollisionModel(os.path.join(grasp.SCENE_DIR, "objects", "bottle_openning.stl"))
        opener.set_homomat(homo(position, grasp.rm.rotmat_from_axangle([0, 0, 1], math.pi / 2)))
        opener.set_rgba([.45, .45, .45, 1.])
        opener.attach_to(base)


def show_animation(plan, actions, speed):
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    base = wd.World(cam_pos=[2., 2.4, 1.7], lookat_pos=[.6, 0., .2])
    gm.gen_frame(length=.3).attach_to(base)
    add_station_models(base, plan)
    plan["target"].set_rgba([.9, .65, .2, 1.])
    for obj in (plan["target"], plan["pipette"]):
        obj.attach_to(base)
    WorkflowAnimation(base, plan, actions, speed)
    base.run()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose-file", help="已有原始 pose.json；默认现场拍照。")
    parser.add_argument("--check-only", action="store_true", help="只规划检查，不打开窗口。")
    parser.add_argument("--speed", type=float, default=1., help="动画播放倍率，不影响真机速度。")
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed 必须为正数")
    pose, pose_file = (grasp.load_grasp_pose(args.pose_file) if args.pose_file
                       else grasp.capture_live_pose())
    print("动画使用配置的双臂起始角，不连接机械臂或夹爪。", flush=True)
    animation_left_start = np.array([
        0.44462849, 0.02638729, 1.59294087,
        1.33750434, 2.58590077, -1.28399963,
    ])
    animation_right_start = np.array([
        -0.76378101, -0.55498721, -1.79941327,
        0.01021574, -2.0996724, 0.40663958,
    ])
    planning_start = perf_counter()
    with grasp._selective_collision_policy():
        plan = grasp.build_grasp_plan(
            animation_left_start, pose, pose_file, animation_right_start)
        build_followup_plan(
            plan, grasp.MOVE_VEL, grasp.MOVE_ACC,
            check_opener_collisions=True,
            opener_target_world_x_offset=0.075,
            validate_grasp_transfer=True,
            optimize_merged_paths=True)
    print("规划耗时：{:.2f} 秒。".format(
        perf_counter() - planning_start), flush=True)
    grasp.print_plan(plan, animation_left_start)
    print(f"完整动作计划已生成：抓取、抬升及后续 {len(plan['followup'])} 步。", flush=True)
    print("开瓶器/移液设备控制未接入；动画展示机械臂动作。参考设备位置需现场核对。", flush=True)
    if not args.check_only:
        actions = animation_actions(plan)
        show_animation(plan, actions, args.speed)


if __name__ == "__main__":
    main()
