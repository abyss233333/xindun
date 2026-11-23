import math
import os
from typing import Literal

import numpy as np

import basis.robot_math as rm
import modeling.collision_model as cm
import modeling.model_collection as mc
from modeling.solid_mesh_collision import containment_contacts
import robot_sim._kinematics.jlchain as jl
# import robot_sim.end_effectors.gripper.dh76.dh76 as hnd
import robot_sim.end_effectors.gripper.PGIA140.PGIA140 as hnd
import robot_sim.manipulators.i5.i5 as rbt
import robot_sim.robots.robot_interface as ri
from trac_ik import TracIK


class I5Dual(ri.RobotInterface):
    """Two I5 arms with fixed workcell stands.

    The class follows the single-arm ``robot_sim.robots.i5.i5`` interface.
    ``lft_arm`` and ``rgt_arm`` each have six joints.  ``both_arm`` accepts
    one 12-element configuration with the left arm first.
    """

    _ARM_DOF = 6
    _DUAL_DOF = 12

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 name="i5_dual", enable_cc=True,
                 enable_cross_arm_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir = os.path.dirname(__file__)
        self.enable_cross_arm_cc = enable_cross_arm_cc
        self._table_edge_x = 0.20
        # The PGIA/Dh50 is mounted with a fixed +90 degree rotation around
        # the arm flange z-axis.  Keep this transform in one place so the
        # initial pose, FK updates, TCP frame, and held-object pose agree.
        self._hand_mount_rotmat = rm.rotmat_from_axangle([0, 0, 1], math.pi / 4*2)

        # Arm mounting offsets are relative to the workcell's global pose.
        self._arm_offsets = {
            "lft_arm": np.array([0.04878976447, 0.13777460, 0.23421023]),
            "rgt_arm": np.array([0.04878976447, -0.13777460, 0.23421023]),
        }
        self._arm_rotmats = {
            "lft_arm": np.dot(np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi/180*-35.26438968),rm.rotmat_from_axangle([1, 0, 0], math.pi/180*-60)),rm.rotmat_from_axangle([0, 0, 1], -math.pi/180*25.52877937)),
            # The right arm is mounted facing the work area.  Keep this as a
            # local arm rotation so the mounting offset is still transformed
            # by the dual robot's global pose.
            "rgt_arm": np.dot(np.dot(rm.rotmat_from_axangle([0, 0, 1], math.pi/180*35.26438968),rm.rotmat_from_axangle([1, 0, 0], math.pi/180*60)),rm.rotmat_from_axangle([0, 0, 1], -math.pi/180*154.47122063)),
        }
        homeconf = np.zeros(self._ARM_DOF)
        lft_homeconf = np.array([0,0,-math.pi/2,0,-math.pi/2,0])
        rgt_homeconf = np.array([0,0,math.pi/2,0,math.pi/2,0])
        self.lft_arm = self._make_arm(
            "lft_arm", pos, rotmat, lft_homeconf, self._arm_offsets["lft_arm"],
            self._arm_rotmats["lft_arm"])
        self.rgt_arm = self._make_arm(
            "rgt_arm", pos, rotmat, rgt_homeconf, self._arm_offsets["rgt_arm"],
            self._arm_rotmats["rgt_arm"])

        self.lft_hnd = hnd.PGIA140(
            pos=self.lft_arm.jnts[-1]['gl_posq'],
            rotmat=np.dot(self.lft_arm.jnts[-1]['gl_rotmatq'], self._hand_mount_rotmat),
            name='lft_hnd', enable_cc=False)
        self.rgt_hnd = hnd.PGIA140(
            pos=self.rgt_arm.jnts[-1]['gl_posq'],
            rotmat=np.dot(self.rgt_arm.jnts[-1]['gl_rotmatq'], self._hand_mount_rotmat),
            name='rgt_hnd', enable_cc=False)

        self.base_stand = jl.JLChain(pos=pos,
                                    rotmat=rotmat,
                                    homeconf=np.zeros(0),
                                    name='base_stand')

        self.base_stand.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "1.STL"),
            cdprimit_type="box", expand_radius=.001)
        self.base_stand.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand.reinitialize()

        self.base_stand2 = jl.JLChain(pos=pos + np.array([0.002956,0.0005,0.31]),
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_stand2')

        self.base_stand2.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "foundation.STL"),
            cdprimit_type="box", expand_radius=.005)
        self.base_stand2.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand2.reinitialize()


        self.base_stand3 = jl.JLChain(pos=pos ,
                                     rotmat=rotmat,
                                     homeconf=np.zeros(0),
                                     name='base_stand3')

        self.base_stand3.lnks[0]['collision_model'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "3.STL"),
            cdprimit_type="box", expand_radius=.005)
        self.base_stand3.lnks[0]['rgba'] = [.35, .35, .35, 1]
        self.base_stand3.reinitialize()

        # self.table_cm = cm.CollisionModel(
        #     os.path.join(this_dir, "meshes", "wholetable.STL"),
        #     cdprimit_type="box", expand_radius=-0.005)
        # table_bmin, table_bmax = self.table_cm.objtrm.bounds
        # self._table_loc_pos = np.array([
        #     self._table_edge_x - table_bmin[0],
        #     -(table_bmin[1] + table_bmax[1]) / 2,
        #     -table_bmax[2],
        # ])
        # self.table = jl.JLChain(
        #     pos=pos + rotmat.dot(self._table_loc_pos),
        #     rotmat=rotmat,
        #     homeconf=np.zeros(0),
        #     name='table')
        # self.table.lnks[0]['collision_model'] = self.table_cm
        # self.table.lnks[0]['rgba'] = [.75, .75, .75, 1]
        # self.table.reinitialize()

        self._set_tcp(self.lft_arm, self.lft_hnd)
        self._set_tcp(self.rgt_arm, self.rgt_hnd)

        # Keep held objects separate so moving one arm cannot move the other
        # arm's objects.
        self.lft_oih_infos = []
        self.rgt_oih_infos = []
        self.iksolver_cache = {}

        if enable_cc:
            self.enable_cc()

        # A hand name is also accepted as a manipulator name for inherited
        # RobotInterface helpers such as cvt_gl_to_loc_tcp.
        self.manipulator_dict.update({
            'lft_arm': self.lft_arm,
            'rgt_arm': self.rgt_arm,
            'lft_hnd': self.lft_arm,
            'rgt_hnd': self.rgt_arm,
        })
        self.hnd_dict.update({
            'lft_arm': self.lft_hnd,
            'rgt_arm': self.rgt_hnd,
            'lft_hnd': self.lft_hnd,
            'rgt_hnd': self.rgt_hnd,
        })

    @staticmethod
    def _make_arm(name, pos, rotmat, homeconf, offset, local_rotmat):
        arm = rbt.i5(
            pos=pos + rotmat.dot(offset),
            rotmat=rotmat.dot(local_rotmat),
            homeconf=homeconf,
            name=name,
            enable_cc=False)
        # This workcell's real I5 arms allow a full turn in either direction.
        # Keep the cached chain ranges/IK limits in sync with the joint data.
        for joint_id in arm.jlc.tgtjnts:
            arm.jlc.jnts[joint_id]['motion_rng'] = np.array([-2 * np.pi, 2 * np.pi])
        arm.jlc.tgtjnts = list(arm.jlc.tgtjnts)
        return arm

    def _set_tcp(self, arm, hand):
        # The hand root is mounted at the arm flange with a fixed rotation.
        # Convert the hand's jaw-center frame into the arm flange frame,
        # which is the frame expected by JLChain when tcp_jnt_id == -1.
        arm.jlc.tcp_jnt_id = -1
        arm.jlc.tcp_loc_pos = self._hand_mount_rotmat.dot(hand.jaw_center_pos)
        arm.jlc.tcp_loc_rotmat = self._hand_mount_rotmat.dot(hand.jaw_center_rotmat)

    def _fix_hand_to_arm(self, arm, hand):
        hand.fix_to(
            pos=arm.jnts[-1]['gl_posq'],
            rotmat=arm.jnts[-1]['gl_rotmatq'].dot(self._hand_mount_rotmat))

    @property
    def _lft_hand_links(self):
        return [self.lft_hnd.lft.lnks[0],
                self.lft_hnd.lft.lnks[1],
                self.lft_hnd.rgt.lnks[1]]

    @property
    def _rgt_hand_links(self):
        return [self.rgt_hnd.lft.lnks[0],
                self.rgt_hnd.lft.lnks[1],
                self.rgt_hnd.rgt.lnks[1]]

    def enable_cc(self):
        if self.cc is not None:
            self.disable_cc()
        super().enable_cc()
        self.cc.add_cdlnks(self.base_stand, [0])
        self.cc.add_cdlnks(self.base_stand2, [0])
        self.cc.add_cdlnks(self.base_stand3, [0])
        self.cc.add_cdlnks(self.lft_arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.rgt_arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.lft_hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.lft_hnd.rgt, [1])
        self.cc.add_cdlnks(self.rgt_hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.rgt_hnd.rgt, [1])

        activelist = [self.base_stand.lnks[0],
                      self.base_stand2.lnks[0],
                      self.base_stand3.lnks[0],
                      self.lft_arm.lnks[0],
                      self.lft_arm.lnks[1],
                      self.lft_arm.lnks[2],
                      self.lft_arm.lnks[3],
                      self.lft_arm.lnks[4],
                      self.lft_arm.lnks[5],
                      self.lft_arm.lnks[6],
                      self.rgt_arm.lnks[0],
                      self.rgt_arm.lnks[1],
                      self.rgt_arm.lnks[2],
                      self.rgt_arm.lnks[3],
                      self.rgt_arm.lnks[4],
                      self.rgt_arm.lnks[5],
                      self.rgt_arm.lnks[6],
                      self.lft_hnd.lft.lnks[0],
                      self.lft_hnd.lft.lnks[1],
                      self.lft_hnd.rgt.lnks[1],
                      self.rgt_hnd.lft.lnks[0],
                      self.rgt_hnd.lft.lnks[1],
                      self.rgt_hnd.rgt.lnks[1]]
        self.cc.set_active_cdlnks(activelist)

        fromlist = [self.base_stand.lnks[0],
                    self.base_stand2.lnks[0],
                    self.base_stand3.lnks[0]]
        intolist = [self.lft_arm.lnks[1],
                    self.lft_arm.lnks[2],
                    self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self.rgt_arm.lnks[1],
                    self.rgt_arm.lnks[2],
                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2],
                    self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[0]]
        intolist = [self.lft_arm.lnks[2],
                    self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[1]]
        intolist = [self.lft_arm.lnks[3],
                    self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[2]]
        intolist = [self.lft_arm.lnks[4],
                    self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[3]]
        intolist = [self.lft_arm.lnks[5],
                    self.lft_arm.lnks[6],
                    self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[4]]
        intolist = [self.lft_arm.lnks[6],
                    self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[5]]
        intolist = [self._lft_hand_links[0],
                    self._lft_hand_links[1],
                    self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.lft_arm.lnks[6]]
        intolist = [self._lft_hand_links[1], self._lft_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[0]]
        intolist = [self.rgt_arm.lnks[2],
                    self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[1]]
        intolist = [self.rgt_arm.lnks[3],
                    self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[2]]
        intolist = [self.rgt_arm.lnks[4],
                    self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[3]]
        intolist = [self.rgt_arm.lnks[5],
                    self.rgt_arm.lnks[6],
                    self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[4]]
        intolist = [self.rgt_arm.lnks[6],
                    self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[5]]
        intolist = [self._rgt_hand_links[0],
                    self._rgt_hand_links[1],
                    self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        fromlist = [self.rgt_arm.lnks[6]]
        intolist = [self._rgt_hand_links[1], self._rgt_hand_links[2]]
        self.cc.set_cdpair(fromlist, intolist)

        if self.enable_cross_arm_cc:
            fromlist = [self.lft_arm.lnks[0],
                        self.lft_arm.lnks[1],
                        self.lft_arm.lnks[2],
                        self.lft_arm.lnks[3],
                        self.lft_arm.lnks[4],
                        self.lft_arm.lnks[5],
                        self.lft_arm.lnks[6],
                        self._lft_hand_links[0],
                        self._lft_hand_links[1],
                        self._lft_hand_links[2]]
            intolist = [self.rgt_arm.lnks[0],
                        self.rgt_arm.lnks[1],
                        self.rgt_arm.lnks[2],
                        self.rgt_arm.lnks[3],
                        self.rgt_arm.lnks[4],
                        self.rgt_arm.lnks[5],
                        self.rgt_arm.lnks[6],
                        self._rgt_hand_links[0],
                        self._rgt_hand_links[1],
                        self._rgt_hand_links[2]]
            self.cc.set_cdpair(fromlist, intolist)

        self._register_existing_held_objects()
        self.last_collision_pairs = []

    def _register_existing_held_objects(self):
        for info in self.lft_oih_infos + self.rgt_oih_infos:
            self._register_held_object(info)

    def _collision_label(self, element):
        for label, chain in (("stand", self.base_stand),
                             ("foundation", self.base_stand2),
                             ("pedestal", self.base_stand3),
                             ("lft_arm", self.lft_arm),
                             ("rgt_arm", self.rgt_arm)):
            for index, link in enumerate(chain.lnks):
                if link is element:
                    return f"{label}/link{index}"
        for label, links in (("lft_hnd", self._lft_hand_links),
                             ("rgt_hnd", self._rgt_hand_links)):
            for name, link in zip(("body", "finger1", "finger2"), links):
                if link is element:
                    return f"{label}/{name}"
        return f"{element.get('arm_name', 'robot')}/held_object"

    @staticmethod
    def _collision_mesh(element):
        model = element['collision_model']
        model.set_pos(element['gl_pos'])
        model.set_rotmat(element['gl_rotmat'])
        return model

    def is_collided(self, obstacle_list=None, otherrobot_list=None,
                    toggle_contact_points=False):
        """Box broad phase followed by mesh checks for registered robot pairs.

        I5 wrist and stand boxes overlap even in valid poses. Confirm those
        candidates using the link meshes instead of omitting whole links.
        External obstacles/robots retain the conservative primitive check.
        """
        if self.cc is None:
            raise RuntimeError("I5 collision checking is disabled")
        self.last_collision_pairs = []
        broad_hit = self.cc.is_collided(
            obstacle_list=[] if obstacle_list is None else obstacle_list,
            otherrobot_list=[] if otherrobot_list is None else otherrobot_list)
        contacts = []
        if broad_hit:
            elements = {self.cc.np.getChild(element['cdprimit_childid']): element
                        for element in self.cc.all_cdelements}
            checked = set()
            for entry in self.cc.chan.getEntries():
                first = elements.get(entry.getFromNodePath())
                second = elements.get(entry.getIntoNodePath())
                pair = frozenset((entry.getFromNodePath(), entry.getIntoNodePath()))
                if pair in checked:
                    continue
                checked.add(pair)
                if first is not None and second is not None:
                    first_model, second_model = self._collision_mesh(first), self._collision_mesh(second)
                    hit, points = first_model.is_mcdwith([second_model], toggle_contacts=True)
                    if not hit:
                        points = containment_contacts(first_model, second_model)
                        if not points:
                            continue
                    labels = (self._collision_label(first), self._collision_label(second))
                    contacts.extend(points)
                else:
                    labels = (self._collision_label(first) if first is not None else "external",
                              self._collision_label(second) if second is not None else "external")
                    contacts.append(np.asarray(entry.getSurfacePoint(self.cc.np)))
                self.last_collision_pairs.append(labels)
                if not toggle_contact_points:
                    return True
        hit = bool(self.last_collision_pairs)
        return (hit, contacts) if toggle_contact_points else hit

    def is_object_collided(self, objcm, contact_hand=None):
        """Check a stationary target, allowing contact only at its fingers.

        The gripper body, flange, both arms and stands remain checked.
        This is also used during open-hand approach and post-place retreat.
        """
        if self.cc is None:
            raise RuntimeError("I5 collision checking is disabled")
        self.last_collision_pairs = []
        fingers = [] if contact_hand is None else (
            self._lft_hand_links[1:] if self._arm_name(contact_hand) == 'lft_arm'
            else self._rgt_hand_links[1:])
        for element in self.cc.all_cdelements:
            if element['collision_model'] is objcm or any(element is link for link in fingers):
                continue
            model = self._collision_mesh(element)
            if model.is_pcdwith(objcm) and (model.is_mcdwith([objcm])
                                           or containment_contacts(model, objcm)):
                self.last_collision_pairs = [(self._collision_label(element), "target_object")]
                return True
        return False

    def _arm_name(self, component_name):
        if component_name in ('lft_arm', 'lft_hnd'):
            return 'lft_arm'
        if component_name in ('rgt_arm', 'rgt_hnd'):
            return 'rgt_arm'
        raise ValueError("The given component or hand name is not supported!")

    def _hand_for_arm(self, arm_name):
        return self.lft_hnd if arm_name == 'lft_arm' else self.rgt_hnd

    def _oih_for_arm(self, arm_name):
        return self.lft_oih_infos if arm_name == 'lft_arm' else self.rgt_oih_infos

    def _register_held_object(self, info):
        arm_name = info['arm_name']
        if self.cc is None:
            return
        if info.get('cdprimit_childid', -1) < 0:
            info['collision_model'].set_pos(info['gl_pos'])
            info['collision_model'].set_rotmat(info['gl_rotmat'])
            into_list = self._held_object_into_list(arm_name)
            registered = self.cc.add_cdobj(
                info['collision_model'], info['rel_pos'], info['rel_rotmat'],
                into_list)
            registered['arm_name'] = arm_name
            infos = self._oih_for_arm(arm_name)
            for index, current in enumerate(infos):
                if current is info:
                    infos[index] = registered
                    break

    def _held_object_into_list(self, arm_name):
        # Only the carrying fingers may contact the object. Include its own
        # arm, flange and hand body, the opposite arm, and both fixed stands.
        stand_links = [self.base_stand.lnks[0], self.base_stand2.lnks[0],
                       self.base_stand3.lnks[0]]
        if arm_name == 'lft_arm':
            return (stand_links + self.lft_arm.lnks[:7] + self._lft_hand_links[:1]
                    + self.rgt_arm.lnks[:7] + self._rgt_hand_links
                    + [info for info in self.rgt_oih_infos
                       if info.get('cdprimit_childid', -1) >= 0])
        return (stand_links + self.rgt_arm.lnks[:7] + self._rgt_hand_links[:1]
                + self.lft_arm.lnks[:7] + self._lft_hand_links
                + [info for info in self.lft_oih_infos
                   if info.get('cdprimit_childid', -1) >= 0])

    def _update_oih(self, arm_name):
        arm = self.lft_arm if arm_name == 'lft_arm' else self.rgt_arm
        for info in self._oih_for_arm(arm_name):
            gl_pos, gl_rotmat = arm.cvt_loc_tcp_to_gl(
                info['rel_pos'], info['rel_rotmat'])
            info['gl_pos'] = gl_pos
            info['gl_rotmat'] = gl_rotmat

    def fix_to(self, pos, rotmat):
        """Move the stands and arms, preserving mounting offsets and joints."""
        self.pos = pos
        self.rotmat = rotmat
        self.base_stand.fix_to(pos=pos, rotmat=rotmat)
        self.base_stand2.fix_to(pos=pos, rotmat=rotmat)
        self.base_stand3.fix_to(pos=pos, rotmat=rotmat)
        for arm_name, arm in (('lft_arm', self.lft_arm),
                              ('rgt_arm', self.rgt_arm)):
            arm.fix_to(pos + rotmat.dot(self._arm_offsets[arm_name]),
                       rotmat.dot(self._arm_rotmats[arm_name]))
            hand = self._hand_for_arm(arm_name)
            self._fix_hand_to_arm(arm, hand)
            self._update_oih(arm_name)

    def _update_component(self, arm_name, jnt_values):
        arm = self.lft_arm if arm_name == 'lft_arm' else self.rgt_arm
        hand = self._hand_for_arm(arm_name)
        status = arm.fk(jnt_values=jnt_values)
        self._fix_hand_to_arm(arm, hand)
        self._update_oih(arm_name)
        return status

    def fk(self, component_name='lft_arm', jnt_values=np.zeros(6)):
        """Run FK for one arm or both arms.

        ``both_arm`` expects ``[left_joints, right_joints]`` flattened into
        one 12-element numpy array.
        """
        if component_name in ('lft_arm', 'lft_hnd', 'rgt_arm', 'rgt_hnd'):
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != self._ARM_DOF:
                raise ValueError("A 1x6 numpy array must be specified for one arm!")
            return self._update_component(self._arm_name(component_name), jnt_values)
        if component_name in ('both_arm', 'both'):
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != self._DUAL_DOF:
                raise ValueError("A 1x12 numpy array must be specified for both arms!")
            lft_status = self._update_component('lft_arm', jnt_values[:6])
            rgt_status = self._update_component('rgt_arm', jnt_values[6:])
            return 'succ' if lft_status == 'succ' and rgt_status == 'succ' else 'out_of_rng'
        raise ValueError("The given component name is not supported!")

    def get_hnd_on_manipulator(self, manipulator_name):
        return self._hand_for_arm(self._arm_name(manipulator_name))

    def tracik(self,  # 定义 TracIK 逆运动学求解接口。
               tgt_pos=np.zeros(3),  # TCP 目标位置，使用世界坐标系下的三维向量。
               tgt_rotmat=np.eye(3),  # TCP 目标姿态，使用世界坐标系下的 3x3 旋转矩阵。
               component_name='lft_arm',  # 指定要求解的机械臂或组件名称。
               urdf_path=os.path.join(os.path.dirname(__file__), "urdf", "i5_dual_427.urdf"),  # 指定 TracIK 使用的机器人 URDF 文件。
               base_link_name='base_link',  # 指定 URDF 中机械臂运动链的基座 link。
               tip_link_name='tool0',  # 指定 URDF 中机械臂运动链的末端 link。
               seed_jnt_values=None,  # 指定 IK 初始关节值；未指定时使用全零向量。
               solver_type: Literal['Speed', 'Distance', 'Manip1', 'Manip2'] = "Distance"):  # 选择 TracIK 的求解策略。
        """在世界坐标系下求解 TCP 目标，并返回关节角解。"""
        arm_name = self._arm_name(component_name)  # 将组件名称规范化为左臂或右臂名称。
        arm = self.lft_arm if arm_name == 'lft_arm' else self.rgt_arm  # 根据名称取得对应机械臂对象。

        # TracIK 的 tip 是法兰处的 tool0；先把模拟器 TCP 目标换算为 URDF tip 目标。
        world_tcp = rm.homomat_from_posrot(tgt_pos, tgt_rotmat)  # 将 TCP 的世界位置和姿态合成为 4x4 齐次矩阵。
        tip_to_tcp = rm.homomat_from_posrot(
            arm.jlc.tcp_loc_pos, arm.jlc.tcp_loc_rotmat)  # 构造 tool0 到 TCP 的局部变换矩阵。
        world_tip = world_tcp @ np.linalg.inv(tip_to_tcp)  # 右乘局部变换的逆矩阵，得到 tool0 的世界目标位姿。
        world_base = rm.homomat_from_posrot(arm.pos, arm.rotmat)  # 构造机械臂基座在世界坐标系中的位姿矩阵。
        base_tip = np.linalg.inv(world_base) @ world_tip  # 将 tool0 世界目标转换到机械臂基座坐标系。
        new_tgt_pos = base_tip[:3, 3]  # 从基座坐标系的齐次矩阵中提取目标位置。
        new_tgt_rot = base_tip[:3, :3]  # 从基座坐标系的齐次矩阵中提取目标旋转矩阵。

        key = (urdf_path, base_link_name, tip_link_name, solver_type)  # 用求解器配置构造缓存键，避免重复创建同类求解器。
        if key not in self.iksolver_cache:  # 缓存中没有该配置的求解器时才创建。
            self.iksolver_cache[key] = TracIK(
                base_link_name=base_link_name,  # 将指定的基座 link 传给求解器。
                tip_link_name=tip_link_name,  # 将指定的末端 link 传给求解器。
                urdf_path=urdf_path,  # 将 URDF 路径传给求解器以载入机器人模型。
                solver_type=solver_type)  # 将求解策略传给新建的 TracIK 实例。
        iksolver = self.iksolver_cache[key]  # 从缓存取得对应配置的 TracIK 求解器。
        if seed_jnt_values is None:  # 调用者没有提供初始关节值时，使用默认 seed。
            seed_jnt_values = np.zeros(self._ARM_DOF)  # 创建长度等于机械臂自由度数的零向量。
        # 求解器可能返回超出 +/-pi 的连续转角；后续会按模拟器的关节范围验证。
        solution = iksolver.ik(new_tgt_pos, new_tgt_rot, seed_jnt_values)  # 根据基座坐标系下的末端目标和初始值执行 IK。
        if solution is None:  # 求解器未找到满足目标的解时，直接返回失败。
            return None  # 用 None 表示本次逆运动学求解失败。
        solution = rm.wrap_to_pi(solution)  # 将各关节角归一化到 [-pi, pi] 区间。
        if (not np.all(np.isfinite(solution))  # 检查解中的每个关节值是否都是有限数。
                or not self.is_jnt_values_in_ranges(arm_name, solution)):  # 拒绝非有限值或超出模拟器关节范围的解。
            return None  # 无效解统一按失败处理。
        return solution  # 返回通过检查的关节角解。

    def get_jnt_values(self, component_name):
        if component_name in ('both_arm', 'both'):
            return np.hstack((self.lft_arm.get_jnt_values(),
                              self.rgt_arm.get_jnt_values()))
        return super().get_jnt_values(component_name)

    def rand_conf(self, component_name):
        if component_name in ('both_arm', 'both'):
            return np.hstack((self.lft_arm.rand_conf(), self.rgt_arm.rand_conf()))
        if component_name in ('lft_arm', 'lft_hnd', 'rgt_arm', 'rgt_hnd'):
            return self.manipulator_dict[self._arm_name(component_name)].rand_conf()
        raise ValueError("The given component name is not supported!")

    def jaw_to(self, hand_name='lft_hnd', jawwidth=0.0,
               both_jawwidth=None):
        if hand_name in ('lft_hnd', 'lft_arm'):
            self.lft_hnd.jaw_to(jawwidth)
            self._set_tcp(self.lft_arm, self.lft_hnd)
        elif hand_name in ('rgt_hnd', 'rgt_arm'):
            self.rgt_hnd.jaw_to(jawwidth)
            self._set_tcp(self.rgt_arm, self.rgt_hnd)
        elif hand_name in ('both', 'both_hnd', 'both_arm'):
            if both_jawwidth is None:
                both_jawwidth = (jawwidth, jawwidth)
            if len(both_jawwidth) != 2:
                raise ValueError("both_jawwidth must contain two jaw widths!")
            self.lft_hnd.jaw_to(both_jawwidth[0])
            self.rgt_hnd.jaw_to(both_jawwidth[1])
            self._set_tcp(self.lft_arm, self.lft_hnd)
            self._set_tcp(self.rgt_arm, self.rgt_hnd)
        else:
            raise ValueError("The given hand name is not supported!")

    def hold(self, hnd_name, objcm, jawwidth=None):
        """Attach an object to the selected hand and collision checker."""
        arm_name = self._arm_name(hnd_name)
        if jawwidth is not None:
            self.jaw_to(hnd_name, jawwidth)
        if self.cc is None:
            raise RuntimeError("Collision checking must be enabled before hold().")
        arm = self.lft_arm if arm_name == 'lft_arm' else self.rgt_arm
        rel_pos, rel_rotmat = arm.cvt_gl_to_loc_tcp(
            objcm.get_pos(), objcm.get_rotmat())
        info = {
            'arm_name': arm_name,
            'collision_model': objcm,
            'rel_pos': rel_pos,
            'rel_rotmat': rel_rotmat,
            'gl_pos': objcm.get_pos(),
            'gl_rotmat': objcm.get_rotmat(),
        }
        self._oih_for_arm(arm_name).append(info)
        self._register_held_object(info)
        return rel_pos, rel_rotmat

    def get_oih_list(self, hnd_name=None):
        if hnd_name is None or hnd_name in ('both', 'both_hnd', 'both_arm'):
            infos = self.lft_oih_infos + self.rgt_oih_infos
        else:
            infos = self._oih_for_arm(self._arm_name(hnd_name))
        result = []
        for info in infos:
            objcm = info['collision_model']
            objcm.set_pos(info['gl_pos'])
            objcm.set_rotmat(info['gl_rotmat'])
            result.append(objcm)
        return result

    def release(self, hnd_name, objcm, jawwidth=None):
        arm_name = self._arm_name(hnd_name)
        if jawwidth is not None:
            self.jaw_to(hnd_name, jawwidth)
        infos = self._oih_for_arm(arm_name)
        for info in list(infos):
            if info['collision_model'] is objcm:
                if self.cc is not None and 'cdprimit_childid' in info:
                    self.cc.delete_cdobj(info)
                infos.remove(info)
                return

    def gen_stickmodel(self, tcp_jnt_id=None, tcp_loc_pos=None,
                       tcp_loc_rotmat=None, toggle_tcpcs=False,
                       toggle_jntscs=False, toggle_connjnt=False,
                       name='i5_dual_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        for stand in (self.base_stand, self.base_stand2):
            stand.gen_stickmodel(
                toggle_tcpcs=False,
                toggle_jntscs=toggle_jntscs,
                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        for arm, hand in ((self.lft_arm, self.lft_hnd),
                          (self.rgt_arm, self.rgt_hnd)):
            arm.gen_stickmodel(
                tcp_jnt_id=tcp_jnt_id,
                tcp_loc_pos=tcp_loc_pos,
                tcp_loc_rotmat=tcp_loc_rotmat,
                toggle_tcpcs=toggle_tcpcs,
                toggle_jntscs=toggle_jntscs,
                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
            # The gripper convenience method references an obsolete ``body``
            # member, so compose its three actual chains here.
            hand.coupling.gen_stickmodel(
                toggle_tcpcs=False,
                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
            hand.lft.gen_stickmodel(
                toggle_tcpcs=False,
                toggle_jntscs=toggle_jntscs,
                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
            hand.rgt.gen_stickmodel(
                toggle_tcpcs=False,
                toggle_jntscs=toggle_jntscs,
                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self, tcp_jnt_id=None, tcp_loc_pos=None,
                      tcp_loc_rotmat=None, toggle_tcpcs=False,
                      toggle_jntscs=False, rgba=None, is_machine=None,
                      is_robot=True, name='i5_dual_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        if is_robot:
            for stand in (self.base_stand, self.base_stand2,self.base_stand3):
                stand.gen_meshmodel(
                    toggle_tcpcs=False,
                    toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
            for arm, hand in ((self.lft_arm, self.lft_hnd),
                              (self.rgt_arm, self.rgt_hnd)):
                arm.gen_meshmodel(
                    tcp_jnt_id=tcp_jnt_id,
                    tcp_loc_pos=tcp_loc_pos,
                    tcp_loc_rotmat=tcp_loc_rotmat,
                    toggle_tcpcs=toggle_tcpcs,
                    toggle_jntscs=toggle_jntscs,
                    rgba=rgba).attach_to(meshmodel)
                hand.gen_meshmodel(
                    toggle_tcpcs=False,
                    toggle_jntscs=toggle_jntscs,
                    rgba=rgba).attach_to(meshmodel)

        for objcm in self.get_oih_list():
            objcm.copy().attach_to(meshmodel)
        return meshmodel


# # Lower-case aliases follow the naming convention of robot_sim.robots.i5.i5.
# i5 = I5Dual
# i5_dual = I5Dual
if __name__ == '__main__':
    import time
    import math
    import sys
    import numpy as np
    import os
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm
    base = wd.World(cam_pos=[4, 3, 1], lookat_pos=[0, 0, 0.0])
    gm.gen_frame().attach_to(base)
    robot = I5Dual(enable_cc=False)

    jnt1 = np.array([ 0.52318408  ,0.14640459 , 1.39730465 , 0.86154571 , 2.34199614, -3.2052807 ])

    jnt2 = np.array([0.230893266889286, 0.1740404736994836, 0.15525519565793824, 0.18783215049444854, 0.18858277167646345, 0.6367832695870834])
    robot.fk("lft_arm", jnt1)
    robot.fk("rgt_arm", jnt2)
    robot.gen_meshmodel().attach_to(base)
    obj = cm.CollisionModel(r"D:\wrs-fujikoshi-main\wrs-fujikoshi-main\xindun_1110\objects\cf.STL")
    obj.set_pos(np.array([0.884, 0.15, -0.045]))
    obj.set_rotmat(rm.rotmat_from_axangle([0, 0, 1], -math.pi / 2))
    obj.attach_to(base)
    base.run()
    a = robot.get_gl_tcp("rgt_arm")
    gm.gen_frame(a[0], a[1]).attach_to(base)
    print(a)
    pos = np.array([-0.5715,  0.5215,  0.1405])
    rotmat = np.array([[-1.0000000e+00,  1.2246468e-16, -1.8369702e-16],
       [ 1.2246468e-16,  1.0000000e+00, -1.8369702e-16],
       [ 1.8369702e-16, -1.8369702e-16, -1.0000000e+00]])

    jnt_values = robot.tracik(pos, rotmat, "rgt_arm")
    print(jnt_values)
    robot.fk("rgt_arm", jnt_values)
    robot.gen_meshmodel().attach_to(base)

    base.run()


    # robot.fk("lft_arm", np.zeros(6))
    # robot.fk("rgt_arm", np.zeros(6))
    #
    # robot.fk("both_arm", np.zeros(12))
    #
    # robot.jaw_to("both", both_jawwidth=(0.02, 0.03))
    #
