"""World-model state vector layout (left arm + left gripper + the blue box).  PRIVILEGED simulator state: it contains the object pose and velocity.
state (34,) = q(8) | dq(8) | prev_ctrl(8) | box_pos(3) | box_quat(4, wxyz) | box_vel(3)
q/dq/prev_ctrl: left joints 1..7 (rad) + left finger1 (m), in actuator order (same as control/action_adapter.py)."""
Q = slice(0, 8); DQ = slice(8, 16); PREV = slice(16, 24); BOX_POS = slice(24, 27); BOX_QUAT = slice(27, 31); BOX_VEL = slice(31, 34); STATE_DIM = 34; ACT_DIM = 8; CONTACT_DIM = 3
PRIVILEGED_SLICES = dict(box_pos=BOX_POS, box_quat=BOX_QUAT, box_vel=BOX_VEL)
GRIPPER_OPEN_M = 0.044
