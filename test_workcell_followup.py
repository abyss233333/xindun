"""Offline checks: no camera, serial port, or robot connection is constructed."""
import ast
import math
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import workcell_followup as followup


class LinearTestRobot:
    """Three coordinates directly controlled by the first three test joints."""
    def __init__(self):
        self.q = np.zeros(6)
        self.lft_oih_infos = []
        self.last_collision_pairs = [('test_arm', 'test_obstacle')]

    def get_jnt_values(self, arm):
        return self.q.copy()

    def get_gl_tcp(self, arm):
        return self.q[:3].copy(), np.eye(3)

    def ik(self, **kwargs):
        return np.r_[kwargs['tgt_pos'], np.zeros(3)]

    def tracik(self, **kwargs):
        return None

    def is_jnt_values_in_ranges(self, arm, q):
        return np.max(np.abs(q)) <= 2 * math.pi

    def fk(self, arm, q):
        self.q = q.copy()
        return 'succ'

    def get_oih_list(self):
        return []

    def is_collided(self, obstacle_list=None):
        return False

    def is_object_collided(self, *args, **kwargs):
        return False


class FollowupTests(unittest.TestCase):
    def test_straight_approach_survives_joining_and_time_interpolation(self):
        class CurvedJointRobot(LinearTestRobot):
            def get_gl_tcp(self, arm):
                q = self.q
                return np.array([q[0] - q[2] ** 2, q[1], q[2]]), np.eye(3)

            def ik(self, **kwargs):
                x, y, z = kwargs['tgt_pos']
                return np.array([x + z ** 2, y, z, 0., 0., 0.])

        robot = CurvedJointRobot()
        robot.q = np.array([.3 ** 2, 0., .3, 0., 0., 0.])
        target = object()
        approach = followup.solve_path(robot, target, dict(
            arm='lft_arm', kind='linear', label='straight approach',
            target=followup.homo([0., 0., .06], np.eye(3))))
        descent = followup.solve_path(robot, target, dict(
            arm='lft_arm', kind='linear', label='descent',
            target=followup.homo([0., 0., 0.], np.eye(3))))
        plan = dict(pregrasp_path=approach, descent=descent)
        joined = followup.build_grasp_approach_path(plan)
        samples = followup.timed_samples(joined, .3, .7)
        # For this kinematic model, a straight line needs curved joint values.
        # Keeping only the endpoints would bow the TCP away from world Z.
        np.testing.assert_allclose(samples[:, 0] - samples[:, 2] ** 2,
                                   0., atol=1e-5)
        np.testing.assert_allclose(samples[:, 1], 0., atol=1e-9)
        np.testing.assert_allclose(samples[:, 3:], 0., atol=1e-9)
        self.assertLessEqual(float(np.max(np.diff(samples[:, 2]))), 1e-8)
        np.testing.assert_allclose(samples[[0, -1], 2], [.3, 0.], atol=1e-9)
        actual_start = approach[0] + 1e-5
        actual_joined = followup.build_grasp_approach_path(plan, actual_start)
        np.testing.assert_array_equal(actual_joined[0], actual_start)
        np.testing.assert_array_equal(actual_joined[1:], joined[1:])
        np.testing.assert_array_equal(approach[0], joined[0])

    def test_joint_paths_join_at_shared_waypoint_only(self):
        first = np.array([[0.] * 6, [.1] * 6])
        second = np.array([[.1] * 6, [.2] * 6, [.3] * 6])
        joined = followup.concatenate_joint_paths(first, second)
        np.testing.assert_allclose(joined, [[0.] * 6, [.1] * 6,
                                            [.2] * 6, [.3] * 6])
        with self.assertRaisesRegex(ValueError, "do not meet"):
            followup.concatenate_joint_paths(first, second + .01)

    def test_selected_followup_steps_merge_into_continuous_paths(self):
        groups = [((11, 12), 0.), ((17, 18, 19), 10.),
                  ((20, 21), 20.), ((22, 23), 30.), ((26, 27), 40.)]
        actions = []
        for steps, offset in groups:
            for index, step in enumerate(steps):
                actions.append(dict(
                    step=step, arm='lft_arm' if step >= 22 else 'rgt_arm',
                    kind='linear', path=np.array([[offset + index] * 6,
                                                  [offset + index + 1] * 6]),
                    samples=np.array([[offset + index] * 6,
                                      [offset + index + 1] * 6]),
                    label=f'step {step}', event='release' if step == 23 else None,
                    target=step, obstacles=[]))
        with patch.object(followup, 'timed_samples', side_effect=lambda path, *_: path.copy()):
            merged = followup.merge_followup_actions(actions, .3, 1.)
        self.assertEqual([action['steps'] for action in merged],
                         [steps for steps, _ in groups])
        for action, (steps, offset) in zip(merged, groups):
            np.testing.assert_allclose(action['path'][:, 0],
                                       np.arange(offset, offset + len(steps) + 1))
            self.assertEqual(action['event'], 'release' if steps[-1] == 23 else None)
            self.assertEqual(action['step'], steps[0])
        step20_21 = next(action for action in merged if action['step'] == 20)
        self.assertEqual(step20_21['steps'], (20, 21))
        self.assertEqual(step20_21['sync_after_step'], 20)
        np.testing.assert_allclose(step20_21['sync_conf'], np.full(6, 21.))
        self.assertEqual(step20_21['sync_sample_index'], 1)

    def test_right_clearance_is_world_negative_y_with_tcp_pointing_down(self):
        for angles in ([.4, -.6, 1.2], [0, math.pi / 2, 0], [math.pi, 0, -.7]):
            start = followup.homo([.5, -.2, .3],
                                  followup.Rotation.from_euler('xyz', angles).as_matrix())
            target = followup.right_clearance_target(start)
            np.testing.assert_allclose(target[:3, 3] - start[:3, 3],
                                       [0, -followup.RIGHT_CLEARANCE_Y, 0], atol=1e-12)
            np.testing.assert_allclose(target[:3, 2], [0, 0, -1], atol=1e-12)
            np.testing.assert_allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(target[:3, :3]), 1.)

    def test_100_point_descent_completes_when_tracik_is_unavailable(self):
        robot = LinearTestRobot()
        action = dict(arm='rgt_arm', kind='linear', label='A descent',
                      target=followup.homo([0, 0, -.1], np.eye(3)))
        with patch.object(robot, 'tracik', side_effect=AssertionError('No timed search needed')):
            path = followup.solve_path(robot, object(), action)
        self.assertEqual(path.shape, (101, 6))
        np.testing.assert_allclose(path[:, 2], np.linspace(0, -.1, 101))

    def test_failed_local_ik_can_use_valid_tracik_fallback(self):
        robot = LinearTestRobot()
        with patch.object(robot, 'ik', return_value=None), patch.object(
                robot, 'tracik', return_value=np.array([.001, 0, 0, 0, 0, 0])):
            q, reason = followup.solve_linear_point(robot, 'rgt_arm',
                np.array([.001, 0, 0]), np.eye(3), np.zeros(6))
        self.assertEqual(reason, '')
        self.assertAlmostEqual(q[0], .001)
        np.testing.assert_array_equal(robot.q, np.zeros(6))

    def test_bad_local_solution_is_rejected_without_relaxing_checks(self):
        for candidate, message in ((np.array([.11, 0, 0, 0, 0, 0]), '关节跳变'),
                                   (np.array([.05, 0, 0, 0, 0, 0]), 'TCP 误差')):
            robot = LinearTestRobot()
            with self.subTest(message=message), patch.object(robot, 'ik', return_value=candidate):
                q, reason = followup.solve_linear_point(robot, 'rgt_arm',
                    np.array([.001, 0, 0]), np.eye(3), np.zeros(6))
                self.assertIsNone(q)
                self.assertIn(message, reason)
                np.testing.assert_array_equal(robot.q, np.zeros(6))

    def test_local_solution_still_requires_collision_check(self):
        robot = LinearTestRobot()
        action = dict(arm='rgt_arm', kind='linear', label='collision',
                      target=followup.homo([.001, 0, 0], np.eye(3)))
        with patch.object(robot, 'is_collided', return_value=True):
            with self.assertRaisesRegex(RuntimeError, '碰撞'):
                followup.solve_path(robot, object(), action)

    def test_external_obstacle_uses_mesh_result_after_bounding_box_overlap(self):
        class Shape:
            def __init__(self, mesh_hit):
                self.mesh_hit = mesh_hit
            def set_pos(self, _position):
                pass
            def set_rotmat(self, _rotation):
                pass
            def is_pcdwith(self, _other):
                return True
            def is_mcdwith(self, _others):
                return self.mesh_hit

        class MeshRobot:
            def __init__(self, mesh_hit):
                self.part = Shape(mesh_hit)
                self.cc = type('CC', (), {
                    'all_cdelements': [dict(collision_model=self.part,
                                            gl_pos=np.zeros(3),
                                            gl_rotmat=np.eye(3))]
                })()
                self.last_collision_pairs = []
            def get_oih_list(self):
                pass
            def is_collided(self, obstacle_list=None, otherrobot_list=None):
                self.last_collision_pairs = []
                return False
            @staticmethod
            def _collision_mesh(element):
                return element['collision_model']
            @staticmethod
            def _collision_label(_element):
                return 'lft_arm/link6'

        obstacle = object.__new__(followup.cm.CollisionModel)
        with patch.object(followup, 'containment_contacts', return_value=[]):
            self.assertFalse(followup.collides_with_obstacles(MeshRobot(False), [obstacle]))
        with patch.object(followup, 'containment_contacts', return_value=[]):
            robot = MeshRobot(True)
            self.assertTrue(followup.collides_with_obstacles(robot, [obstacle]))
            self.assertEqual(robot.last_collision_pairs,
                             [('lft_arm/link6', 'external_obstacle/0')])

    def test_ignored_held_object_table_pair_keeps_robot_table_check(self):
        class Shape:
            def __init__(self, mesh_hit, broad_hit=True):
                self.mesh_hit, self.broad_hit = mesh_hit, broad_hit
            def set_pos(self, _position):
                pass
            def set_rotmat(self, _rotation):
                pass
            def is_pcdwith(self, _other):
                return self.broad_hit
            def is_mcdwith(self, _others):
                return self.mesh_hit

        held_object = Shape(True)
        arm_link = Shape(False, broad_hit=False)
        class Robot:
            def __init__(self):
                self.cc = type('CC', (), {'all_cdelements': [
                    dict(collision_model=held_object, label='lft_arm/held_object',
                         gl_pos=np.zeros(3), gl_rotmat=np.eye(3)),
                    dict(collision_model=arm_link, label='lft_arm/link6',
                         gl_pos=np.zeros(3), gl_rotmat=np.eye(3)),
                ]})()
                self.last_collision_pairs = []
            def get_oih_list(self):
                pass
            def is_collided(self, obstacle_list=None, otherrobot_list=None):
                return False
            @staticmethod
            def _collision_mesh(element):
                return element['collision_model']
            @staticmethod
            def _collision_label(element):
                return element['label']

        obstacle = object.__new__(followup.cm.CollisionModel)
        robot = Robot()
        ignored = {('lft_arm/held_object', 0)}
        self.assertFalse(followup.collides_with_obstacles(
            robot, [obstacle], ignore_obstacle_pairs=ignored))
        arm_link.broad_hit = arm_link.mesh_hit = True
        self.assertTrue(followup.collides_with_obstacles(
            robot, [obstacle], ignore_obstacle_pairs=ignored))
        self.assertEqual(robot.last_collision_pairs,
                         [('lft_arm/link6', 'external_obstacle/0')])

    def test_table_skips_arm_and_hand_pairs_but_checks_held_bottle(self):
        class Shape:
            def set_pos(self, _position):
                pass
            def set_rotmat(self, _rotation):
                pass
            def is_pcdwith(self, _other):
                return True
            def is_mcdwith(self, _others):
                return True

        class Robot:
            def __init__(self, label):
                self.part = Shape()
                self.cc = type('CC', (), {'all_cdelements': [dict(
                    collision_model=self.part, label=label,
                    gl_pos=np.zeros(3), gl_rotmat=np.eye(3))]})()
                self.last_collision_pairs = []
            def get_oih_list(self):
                pass
            def is_collided(self, obstacle_list=None, otherrobot_list=None):
                return False
            @staticmethod
            def _collision_mesh(element):
                return element['collision_model']
            @staticmethod
            def _collision_label(element):
                return element['label']

        table = object.__new__(followup.cm.CollisionModel)
        table._xindun_table_obstacle = True
        for label in ('lft_arm/link3', 'rgt_arm/link6',
                      'lft_hnd/finger1', 'rgt_hnd/body'):
            with self.subTest(label=label):
                self.assertFalse(followup.collides_with_obstacles(
                    Robot(label), [table]))
        bottle = Robot('lft_arm/held_object')
        self.assertTrue(followup.collides_with_obstacles(bottle, [table]))
        self.assertEqual(bottle.last_collision_pairs,
                         [('lft_arm/held_object', 'external_obstacle/0')])

    def test_return_path_interpolates_rotation_and_keeps_tcp_on_line(self):
        class PoseRobot(LinearTestRobot):
            def get_gl_tcp(self, arm):
                return self.q[:3].copy(), followup.Rotation.from_rotvec(self.q[3:]).as_matrix()

            def ik(self, **kwargs):
                return np.r_[kwargs['tgt_pos'],
                             followup.Rotation.from_matrix(kwargs['tgt_rotmat']).as_rotvec()]

        robot = PoseRobot()
        goal = followup.homo([.12, -.08, .06],
                            followup.rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2))
        action = dict(arm='lft_arm', kind='linear', label='return', target=goal,
                      interpolate_rotation=True)
        path = followup.solve_path(robot, object(), action)
        np.testing.assert_allclose(path[:, :3], np.linspace(np.zeros(3), goal[:3, 3], len(path)))
        np.testing.assert_allclose(path[-1, 3:], [0, 0, -math.pi / 2], atol=1e-12)
        self.assertLessEqual(np.max(np.linalg.norm(np.diff(path[:, 3:], axis=0), axis=1)),
                             followup.LINEAR_ANGULAR_SPACING + 1e-12)
        robot = PoseRobot()
        action['interpolate_rotation'] = False
        with self.assertRaisesRegex(RuntimeError, '起终点姿态不一致'):
            followup.solve_path(robot, object(), action)

    def test_linear_path_preserves_turn_across_pi_boundary(self):
        class Robot:
            lft_oih_infos = []
            last_collision_pairs = []
            q = np.array([3.14, 0, 0, 0, 0, 0.])
            def get_jnt_values(self, arm):
                return self.q.copy()
            def get_gl_tcp(self, arm):
                x = (self.q[0] - 3.14) / (math.pi + .001 - 3.14) * .001
                return np.array([x, 0, 0]), np.eye(3)
            def tracik(self, **kwargs):
                return np.array([-math.pi + .001, 0, 0, 0, 0, 0.])
            def ik(self, **kwargs):
                return None
            def is_jnt_values_in_ranges(self, arm, q):
                return np.max(np.abs(q)) < 2 * math.pi
            def fk(self, arm, q):
                self.q = q.copy()
                return 'succ'
            def get_oih_list(self):
                return []
            def is_collided(self, obstacle_list=None):
                return False
            def is_object_collided(self, *args, **kwargs):
                return False
        action = dict(arm='rgt_arm', kind='linear', label='cross pi',
                      target=followup.homo([.001, 0, 0], np.eye(3)))
        path = followup.solve_path(Robot(), object(), action)
        self.assertGreater(path[-1, 0], math.pi)
        self.assertLess(np.max(np.abs(path[-1] - path[0])), .01)

    def test_table_top_is_two_millimetres_below_transformed_object_bottom(self):
        vertices = np.array([[0., 0., -.04], [.02, 0., .03],
                             [0., .01, .01]])
        target = type('Target', (), {
            'objtrm': type('Mesh', (), {'vertices': vertices})(),
        })()
        bounds = np.array([[-1.25, 0., -.75], [1.25, 1.5, 0.]])
        table = type('Table', (), {
            'objtrm': type('Mesh', (), {'bounds': bounds})(),
            'set_pos': lambda self, pos: setattr(self, 'position', np.asarray(pos)),
            'set_rotmat': lambda self, rot: setattr(self, 'rotation', np.asarray(rot)),
            'set_rgba': lambda self, rgba: None,
        })()
        pose = followup.homo([.70, .22, -.038],
                             followup.rm.rotmat_from_axangle([0, 1, 0], .4))
        expected_world = vertices @ pose[:3, :3].T + pose[:3, 3]
        with patch.object(followup.cm, 'CollisionModel', return_value=table):
            result = followup.make_table_obstacle(pose, target)
        self.assertIs(result, table)
        np.testing.assert_allclose(
            result.position[:2]
            + result.rotation[:2, :2] @ np.mean(bounds[:, :2], axis=0),
            pose[:2, 3] + followup.TABLE_OFFSET_WORLD_XY)
        np.testing.assert_allclose(result.rotation @ [1., 0., 0.], [0., 1., 0.],
                                   atol=1e-12)
        self.assertAlmostEqual(
            result.position[2] + bounds[1, 2],
            np.min(expected_world[:, 2]) - .003,
            places=12)

    def test_stations_stay_fixed_but_placement_follows_rotated_live_pose(self):
        grasp = followup.homo([0, 0, .025], followup.rm.rotmat_from_axangle([1, 0, 0], math.pi))
        def plan_at(pos, angle):
            pose = followup.homo(pos, followup.rm.rotmat_from_axangle([0, 0, 1], angle))
            return dict(object_pose=pose, jaw_to_object=grasp,
                        pregrasp_pos=(pose @ grasp)[:3, 3] + [0, 0, .06],
                        table_obstacle=object())
        first, second = plan_at([.9, .18, -.03], -.7), plan_at([.87, .12, -.02], -1.2)
        a, b = followup.station_actions(first), followup.station_actions(second)
        preview = followup.station_actions(first, check_opener_collisions=False)
        self.assertEqual([item['step'] for item in a], list(range(5, 30)))
        self.assertTrue(all(item['kind'] != 'joint' for item in a))
        self.assertEqual([len(item['obstacles']) for item in preview
                          if item['step'] in (5, 8, 27)], [1, 1, 1])
        for left, right in zip(a, b):
            if left['step'] not in range(22, 27):
                np.testing.assert_allclose(left['target'], right['target'])
        targets = {item['step']: item for item in b}
        np.testing.assert_allclose(targets[23]['target'], second['object_pose'] @ grasp)
        np.testing.assert_allclose(targets[25]['target'], targets[23]['target'])
        self.assertEqual(targets[22]['kind'], 'linear')
        self.assertTrue(targets[22]['interpolate_rotation'])
        self.assertEqual(targets[23]['event'], 'release')
        for step in (22, 23, 24, 25, 26, 27):
            expected_ignored_pairs = (
                [('lft_arm/held_object', 0)]
                if step in (23, 24, 25, 26) else [])
            self.assertEqual(targets[step]['ignore_obstacle_pairs'],
                             expected_ignored_pairs)
        self.assertEqual(targets[25]['event'], 'hold')

        # Compare classifications without importing the scene-loading script.
        reference = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                 'xindun_1110-2', 'plan_linear_actions.py')
        with open(reference, encoding='utf-8') as stream:
            module = ast.parse(stream.read())
        linear_steps = next(ast.literal_eval(node.value) for node in module.body
                            if isinstance(node, ast.Assign) and any(
                                isinstance(t, ast.Name) and t.id == 'LINEAR_STEPS'
                                for t in node.targets))
        self.assertEqual({item['step'] for item in a if item['kind'] == 'linear'},
                         linear_steps.intersection(range(5, 30)))

        for step in (5, 27):
            action = next(item for item in b if item['step'] == step)
            self.assertEqual((action['kind'], action['planner']), ('path', 'rrtc'))
            self.assertEqual(len(action['obstacles']), 4)
            self.assertNotIn('goal_joints', action)
            bottle = action['target'] @ np.linalg.inv(grasp)
            np.testing.assert_allclose(bottle[:3, 3], [.9, .2, .20])
            np.testing.assert_allclose(action['target'][:3, :3] @ [0, 0, 1],
                                       [1, 0, 0], atol=1e-12)

        step7 = next(item for item in b if item['step'] == 7)
        step8 = next(item for item in b if item['step'] == 8)
        self.assertEqual(step7['kind'], 'linear')
        self.assertEqual((step8['kind'], step8['planner']), ('path', 'rrtc'))
        self.assertEqual(len(step8['obstacles']), 4)
        step7_bottle = step7['target'] @ np.linalg.inv(grasp)
        np.testing.assert_allclose(step7_bottle[:3, 3], [.9, .2, .20])
        self.assertIsNone(step7['event'])

        # Recover the bottle pose from the jaw target, including grasp rotation.
        for action in (item for item in b if item['step'] in (5, 6, 7, 27, 28, 29)):
            step = action['step']
            bottle = action['target'] @ np.linalg.inv(grasp)
            np.testing.assert_allclose(bottle[:3, :3] @ [1, 0, 0], [0, 0, 1], atol=1e-12)
            np.testing.assert_allclose(action['target'][:3, :3] @ [0, 0, 1],
                                       [1, 0, 0], atol=1e-12)
            np.testing.assert_allclose(bottle[:3, 3],
                                       [.9, .2, .235 if step in (6, 28) else .20])
        bottle_work = targets[8]['target'] @ np.linalg.inv(grasp)
        np.testing.assert_allclose(bottle_work[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-12)
        np.testing.assert_allclose(bottle_work[:3, :3] @ [0, 0, 1], [0, 0, 1], atol=1e-12)
        np.testing.assert_allclose(bottle_work[:3, 3], [.6, 0, .2])

    def make_execution(self, fail=False):
        events = []
        class Arm:
            def __init__(self, name):
                self.name, self.q = name, np.zeros(6)
            def get_jnt_values(self):
                return self.q.copy()
            def get_motion_status(self):
                return {'steady': True}
            def _check_motion_ready(self, status):
                pass
            def prepare_jntspace_path_buffer(self, samples, **kwargs):
                events.append((self.name, 'prepare', kwargs))
                return {'state': 'prepared', 'samples': np.asarray(samples).copy()}
            def release_prepared_jntspace_path_buffer(self, buffer):
                buffer['state'] = 'released'
            def move_jntspace_path(self, path, **kwargs):
                events.append((self.name, 'path', dict(kwargs, path=np.asarray(path).copy())))
                if fail:
                    raise RuntimeError('rejected servo path')
                self.q = np.array(path[-1])
                buffer = kwargs.get('prepared_buffer')
                if buffer is not None:
                    buffer['state'] = 'released'
            def stop(self):
                events.append((self.name, 'stop', {}))
        class Gripper:
            def open_g(self):
                events.append(('gripper', 'open', {}))
            def close_g(self):
                events.append(('gripper', 'close', {}))
        plan = dict(target=object(), descent=[np.zeros(6)], right_start=np.zeros(6), right_wait=np.zeros(6), followup=[
            dict(step=5, arm='lft_arm', kind='path', path=np.array([[0]*6, [.1]*6]),
                 samples=np.array([[0]*6, [.1]*6]), label='left transfer', event=None),
            dict(step=10, arm='rgt_arm', kind='linear', path=np.array([[0]*6, [.2]*6]),
                 samples=np.array([[0]*6, [.1]*6, [.2]*6]), label='right insertion', event=None),
            dict(step=23, arm='lft_arm', kind='linear', path=np.array([[.1]*6, [0]*6]),
                 samples=np.array([[.1]*6, [0]*6]), label='place', event='release'),
            dict(step=25, arm='lft_arm', kind='linear', path=np.array([[0]*6, [.1]*6]),
                 samples=np.array([[0]*6, [.1]*6]), label='regrasp', event='hold')])
        return events, plan, Arm('left'), Arm('right'), Gripper()

    def make_clearance_execution(self):
        events, plan, left, right, gripper = self.make_execution()
        plan.update(start_joints=np.zeros(6),
                    hover_path=np.array([[0.] * 6, [.05] * 6]),
                    pregrasp_path=np.array([[.05] * 6, [.075] * 6, [.1] * 6]),
                    descent=np.array([[.1] * 6, [.12] * 6]),
                    right_wait=np.full(6, .25),
                    right_clearance=dict(path=np.array([[0.] * 6, [.25] * 6])))
        return events, plan, left, right, gripper

    def test_grasp_moves_high_and_right_clearance_concurrently_then_opens(self):
        from threading import Barrier
        import real_grasp as grasp
        events, plan, left, right, gripper = self.make_clearance_execution()
        start_barrier = Barrier(2)
        left_move, right_move = left.move_jntspace_path, right.move_jntspace_path
        def parallel_left(path, **kwargs):
            if kwargs.get('prepared_buffer') is not None:
                events.append(('left', 'parallel_start', {}))
                start_barrier.wait(timeout=2.)
                left_move(path, **kwargs)
                events.append(('left', 'parallel_done', {}))
            else:
                left_move(path, **kwargs)
        def parallel_right(path, **kwargs):
            if kwargs.get('prepared_buffer') is not None:
                events.append(('right', 'parallel_start', {}))
                start_barrier.wait(timeout=2.)
                right_move(path, **kwargs)
                events.append(('right', 'parallel_done', {}))
            else:
                right_move(path, **kwargs)
        left.move_jntspace_path = parallel_left
        right.move_jntspace_path = parallel_right
        grasp.execute_grasp(plan, left, gripper, right)
        names = [(arm_name, kind) for arm_name, kind, _ in events]
        starts = [names.index(('left', 'parallel_start')),
                  names.index(('right', 'parallel_start'))]
        completes = [names.index(('left', 'parallel_done')),
                     names.index(('right', 'parallel_done'))]
        self.assertLess(max(starts), min(completes))
        self.assertGreater(names.index(('gripper', 'open')), max(completes))
        self.assertEqual(names[-1], ('gripper', 'close'))
        left_paths = [kwargs['path'] for arm_name, kind, kwargs in events
                      if arm_name == 'left' and kind == 'path']
        np.testing.assert_allclose(left_paths[0], [[0.] * 6, [.05] * 6])
        np.testing.assert_allclose(left_paths[1], [[.05] * 6, [.075] * 6, [.1] * 6,
                                                   [.12] * 6])
        np.testing.assert_array_equal(right.q, plan['right_wait'])
        np.testing.assert_array_equal(left.q, plan['descent'][-1])

    def test_failed_right_clearance_stops_before_left_approach(self):
        import real_grasp as grasp
        events, plan, left, right, gripper = self.make_clearance_execution()
        with patch.object(right, 'move_jntspace_path', side_effect=RuntimeError('right path rejected')):
            with self.assertRaisesRegex(RuntimeError, 'right path rejected'):
                grasp.execute_grasp(plan, left, gripper, right)
        names = [(arm_name, kind) for arm_name, kind, _ in events]
        self.assertIn(('left', 'path'), names)
        self.assertIn(('left', 'stop'), names)
        self.assertIn(('right', 'stop'), names)
        self.assertNotIn(('gripper', 'open'), names)

    def test_preview_inserts_right_clearance_after_open_before_pregrasp(self):
        import preview_live_workflow as preview
        _, plan, _, _, _ = self.make_clearance_execution()
        plan['followup'][0]['path'] = np.array([[.1] * 6, [.2] * 6])
        plan['followup'][0]['samples'] = plan['followup'][0]['path'].copy()
        with patch.object(preview, 'timed_samples', side_effect=lambda p, *args: np.asarray(p)):
            actions = preview.animation_actions(plan)
        initial = actions[0]
        self.assertEqual([item['arm'] for item in initial['parallel_actions']],
                         ['lft_arm', 'rgt_arm'])
        self.assertEqual([item['event'] for item in initial['parallel_actions']],
                         ['open', None])
        self.assertEqual(len(initial['parallel_samples']), 2)
        np.testing.assert_array_equal(initial['parallel_actions'][1]['path'][0],
                                      plan['right_start'])
        np.testing.assert_array_equal(initial['parallel_actions'][1]['path'][-1],
                                      plan['right_wait'])
        np.testing.assert_allclose(actions[1]['path'], [[.05] * 6, [.075] * 6, [.1] * 6,
                                                        [.12] * 6])
        np.testing.assert_allclose(actions[2]['path'], [[.12] * 6, [.1] * 6, [.2] * 6])

    def test_followup_merges_lift_and_step5_into_one_buffered_path(self):
        events, plan, left, right, gripper = self.make_clearance_execution()
        left.q = plan['descent'][-1].copy()
        right.q = plan['right_wait'].copy()
        step5 = dict(plan['followup'][0],
                     path=np.array([[.1] * 6, [.2] * 6], dtype=float),
                     samples=np.array([[.1] * 6, [.2] * 6], dtype=float))
        plan['followup'] = [step5]
        followup.execute_followup(plan, left, right, gripper)
        paths = [kwargs['path'] for arm_name, kind, kwargs in events
                 if arm_name == 'left' and kind == 'path']
        self.assertEqual(len(paths), 1)
        np.testing.assert_allclose(paths[0], [[.12] * 6, [.1] * 6, [.2] * 6])

    def test_followup_checks_parked_right_position_instead_of_original_start(self):
        _, plan, left, right, gripper = self.make_execution()
        right.q = np.full(6, .25)
        plan['right_wait'] = right.q.copy()
        plan['followup'][1]['path'][0] = right.q
        plan['followup'][1]['samples'][0] = right.q
        with patch.object(followup.time, 'monotonic', side_effect=range(100)), patch.object(followup.time, 'sleep'):
            followup.execute_followup(plan, left, right, gripper)

    def test_dual_arm_commands_and_gripper_events_run_in_order(self):
        events, plan, left, right, gripper = self.make_execution()
        with patch.object(followup.time, 'sleep') as sleep:
            followup.execute_followup(plan, left, right, gripper, vel=.3, acc=1.)
        sleep.assert_not_called()
        self.assertEqual([(a, b) for a, b, _ in events], [
            ('left', 'path'), ('right', 'path'), ('left', 'path'),
            ('gripper', 'open'), ('left', 'path'), ('gripper', 'close')])
        self.assertFalse(any(kind == 'joint' for _, kind, _ in events))
        for _, kind, kwargs in events:
            if kind == 'path':
                self.assertEqual(kwargs['execution_mode'], 'buffer')
                self.assertEqual(kwargs['vel'], .3)
                self.assertIn('prepared_samples', kwargs)

    def test_steps_8_and_9_start_together_and_step_10_waits_for_both(self):
        from threading import Barrier
        events, plan, left, right, gripper = self.make_execution()
        def action(step, arm, start, end):
            path = np.asarray([np.full(6, start), np.full(6, end)])
            return dict(step=step, arm=arm, path=path,
                        samples=np.asarray([np.full(6, start),
                                            np.full(6, (start + end) / 2),
                                            np.full(6, end)]),
                        label=f'step {step}', event=None)
        plan['followup'] = [
            action(8, 'lft_arm', 0., .1),
            action(9, 'rgt_arm', 0., .2),
            action(10, 'rgt_arm', .2, .3),
        ]
        barrier = Barrier(2)
        def left_move(path, **kwargs):
            events.append(('left', 'parallel_start', {}))
            barrier.wait(timeout=2.)
            left.q = np.asarray(path[-1]).copy()
            events.append(('left', 'parallel_done', {}))
        right_calls = [0]
        def right_move(path, **kwargs):
            right_calls[0] += 1
            if right_calls[0] == 1:
                events.append(('right', 'parallel_start', {}))
                barrier.wait(timeout=2.)
                right.q = np.asarray(path[-1]).copy()
                events.append(('right', 'parallel_done', {}))
            else:
                events.append(('right', 'step10_start', {}))
                right.q = np.asarray(path[-1]).copy()
        left.move_jntspace_path = left_move
        right.move_jntspace_path = right_move

        followup.execute_followup(plan, left, right, gripper)

        names = [(arm, event) for arm, event, _ in events]
        started = [names.index(('left', 'parallel_start')),
                   names.index(('right', 'parallel_start'))]
        completed = [names.index(('left', 'parallel_done')),
                     names.index(('right', 'parallel_done'))]
        self.assertLess(max(started), min(completed))
        self.assertGreater(names.index(('right', 'step10_start')), max(completed))

    def test_left_steps_22_to_24_start_at_the_step20_waypoint_during_merged_right_path(self):
        from threading import Event
        events, plan, left, right, gripper = self.make_execution()
        right_path = np.asarray([[0.] * 6, [.1] * 6, [.2] * 6])
        left_return = np.asarray([[0.] * 6, [.1] * 6, [.2] * 6])
        left_lift = np.asarray([[.2] * 6, [.3] * 6])
        plan['followup'] = [
            dict(step=20, steps=(20, 21), sync_after_step=20,
                 sync_conf=np.full(6, .1), sync_sample_index=1,
                 arm='rgt_arm', kind='path', path=right_path,
                 samples=right_path.copy(), label='right steps 20-21', event=None),
            dict(step=22, steps=(22, 23), arm='lft_arm', kind='linear',
                 path=left_return, samples=left_return.copy(),
                 label='left steps 22-23', event='release'),
            dict(step=24, arm='lft_arm', kind='linear', path=left_lift,
                 samples=left_lift.copy(), label='left step 24', event=None),
        ]
        at_boundary, continue_right = Event(), Event()

        def right_move(path, **kwargs):
            events.append(('right', 'right_path_started', {}))
            right.q = np.full(6, .1)
            at_boundary.set()
            if not continue_right.wait(2.):
                raise TimeoutError('left step 24 did not start')
            right.q = np.asarray(path[-1]).copy()
            events.append(('right', 'right_path_finished', {}))
            kwargs['prepared_buffer']['state'] = 'released'

        left_calls = [0]
        def left_move(path, **kwargs):
            left_calls[0] += 1
            if left_calls[0] == 1:
                if not at_boundary.wait(2.):
                    raise TimeoutError('right arm did not reach step 20 waypoint')
                events.append(('left', 'steps22_23_started', {}))
            else:
                events.append(('left', 'step24_started', {}))
            left.q = np.asarray(path[-1]).copy()
            kwargs['prepared_buffer']['state'] = 'released'
            if left_calls[0] == 2:
                continue_right.set()

        right.move_jntspace_path = right_move
        left.move_jntspace_path = left_move
        followup.execute_followup(plan, left, right, gripper)

        names = [(arm, event) for arm, event, _ in events]
        self.assertLess(names.index(('right', 'right_path_started')),
                         names.index(('left', 'steps22_23_started')))
        self.assertLess(names.index(('left', 'steps22_23_started')),
                         names.index(('left', 'step24_started')))
        self.assertLess(names.index(('left', 'step24_started')),
                         names.index(('right', 'right_path_finished')))
        self.assertLess(names.index(('gripper', 'open')),
                         names.index(('left', 'step24_started')))

    def test_preview_groups_steps_8_and_9_as_one_parallel_action(self):
        import preview_live_workflow as preview
        _, plan, _, _, _ = self.make_clearance_execution()
        def action(step, arm):
            samples = np.asarray([[0.] * 6, [.1] * 6])
            return dict(step=step, arm=arm, path=samples, samples=samples,
                        kind='path', event=None, label=f'step {step}')
        plan['followup'] = [action(8, 'lft_arm'), action(9, 'rgt_arm'),
                            action(10, 'rgt_arm')]
        actions = preview.animation_actions(plan)
        parallel = next(item for item in actions if 'parallel_actions' in item)
        self.assertEqual([item['arm'] for item in parallel['parallel_actions']],
                         ['lft_arm', 'rgt_arm'])
        self.assertEqual(len(parallel['parallel_samples']), 2)
        self.assertEqual(len(parallel['parallel_samples'][0]),
                         len(parallel['parallel_samples'][1]))

    def test_preview_starts_left_return_at_step20_and_keeps_right_path_continuous(self):
        import preview_live_workflow as preview
        _, plan, _, _, _ = self.make_clearance_execution()
        right_path = np.asarray([[0.] * 6, [.2] * 6])
        right_samples = np.repeat(np.arange(10, dtype=float)[:, None], 6, axis=1)
        left_return = np.asarray([[0.] * 6, [.1] * 6, [.2] * 6])
        left_lift = np.asarray([[.2] * 6, [.3] * 6])
        plan['followup'] = [
            dict(step=20, steps=(20, 21), sync_after_step=20,
                 sync_conf=np.full(6, 2.), sync_sample_index=2,
                 arm='rgt_arm', kind='path', path=right_path,
                 samples=right_samples, label='right steps 20-21', event=None),
            dict(step=22, steps=(22, 23), arm='lft_arm', kind='linear',
                 path=left_return, samples=np.asarray([[0.] * 6, [.1] * 6, [.2] * 6]),
                 label='left steps 22-23', event='release'),
            dict(step=24, arm='lft_arm', kind='linear', path=left_lift,
                 samples=left_lift.copy(), label='left step 24', event=None),
        ]
        actions = preview.animation_actions(plan)
        timeline = next(item for item in actions if 'indexed_events' in item)
        self.assertEqual(timeline['indexed_events'], [(4, 'release')])
        self.assertEqual(len(timeline['parallel_samples'][0]),
                         len(timeline['parallel_samples'][1]))
        np.testing.assert_allclose(timeline['parallel_samples'][1][:3],
                                   np.zeros((3, 6)))
        np.testing.assert_allclose(timeline['parallel_samples'][0][:10, 0],
                                   np.arange(10, dtype=float))

        animation = preview.WorkflowAnimation.__new__(preview.WorkflowAnimation)
        animation.actions = [timeline]
        animation.stride = 2
        animation._prepare_release_holds()
        left_timeline = timeline['parallel_samples'][1]
        right_timeline = timeline['parallel_samples'][0]
        np.testing.assert_allclose(left_timeline[5:29], np.full((24, 6), .2))
        self.assertGreater(right_timeline[5, 0], right_timeline[4, 0])

    def test_path_failure_stops_both_arms_without_later_gripper_events(self):
        events, plan, left, right, gripper = self.make_execution(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'rejected servo path'):
            followup.execute_followup(plan, left, right, gripper)
        self.assertEqual([(a, b) for a, b, _ in events], [
            ('left', 'path'), ('left', 'stop'), ('right', 'stop')])

    def test_wrong_actual_start_does_not_send_motion(self):
        events, plan, left, right, gripper = self.make_execution()
        right.q[:] = .5
        with patch.object(followup.time, 'monotonic', side_effect=range(100)):
            with patch.object(followup.time, 'sleep'):
                with self.assertRaisesRegex(RuntimeError, '未确认实际到位'):
                    followup.execute_followup(plan, left, right, gripper)
        self.assertEqual([(a, b) for a, b, _ in events], [('left', 'stop'), ('right', 'stop')])


if __name__ == '__main__':
    unittest.main()
