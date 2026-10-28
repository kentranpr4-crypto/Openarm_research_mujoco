"""Camera definitions + offscreen RGB / depth / segmentation rendering for the OpenArm lab scene.

Camera names are part of the dataset contract (policy inputs use FRONT + WRIST_*; OVERVIEW is for humans only).
Mount frames (verified from the model, see docs): in `openarm_<side>_hand`, local +z = fingers' pointing direction
(= world +x at the home pose), local +x = world up, local y = world -y. MuJoCo cameras look along their own -z, so the
wrist camera uses xyaxes = (local y, tilted local +x).  The D435-like head camera spec values come from memory and are
NOT checked against the Intel datasheet."""
import numpy as np, mujoco

OVERVIEW, FRONT, WRIST_LEFT, WRIST_RIGHT = "overview", "front", "wrist_left", "wrist_right"
POLICY_CAMERAS = (FRONT, WRIST_LEFT, WRIST_RIGHT)
ALL_CAMERAS = (OVERVIEW, FRONT, WRIST_LEFT, WRIST_RIGHT)

HEAD_FOVY = 42.5                  # D435-like colour vertical FOV [deg] (unverified)
HEAD_POS, HEAD_PITCH = (0.06, 0.0, 0.80), 35.0       # in openarm_body_link0, pitched down [deg]
WRIST_FOVY, WRIST_PITCH = 90.0, 35.0                 # wide wrist cam, tilted down so BOTH fingertips are in view (checked visually)
WRIST_POS = (0.04, 0.0, -0.02)                       # hand frame: 4 cm above the flange axis (hand housing top is x=0.029), 2 cm behind it
OVERVIEW_POS, OVERVIEW_LOOKAT = (1.55, -1.05, 1.05), (0.38, 0.02, 0.42)

def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    z = np.asarray(pos, float) - np.asarray(target, float); z /= np.linalg.norm(z)   # camera looks along -z
    x = np.cross(up, z); x /= np.linalg.norm(x); y = np.cross(z, x)
    return [*x, *y]

def add_cameras(spec, wrist=True):
    """Add overview (world), front (torso head), wrist_left/right (hand frames). Cameras carry no geometry, so they
    cannot collide with or visually block anything."""
    spec.worldbody.add_camera(name=OVERVIEW, pos=list(OVERVIEW_POS), xyaxes=lookat_xyaxes(OVERVIEW_POS, OVERVIEW_LOOKAT), fovy=45.0)
    p = np.radians(HEAD_PITCH)
    spec.body("openarm_body_link0").add_camera(name=FRONT, pos=list(HEAD_POS), xyaxes=[0, -1, 0, np.sin(p), 0, np.cos(p)], fovy=HEAD_FOVY)
    if wrist:
        t = np.radians(WRIST_PITCH)
        for side, name in (("left", WRIST_LEFT), ("right", WRIST_RIGHT)):
            spec.body(f"openarm_{side}_hand").add_camera(name=name, pos=list(WRIST_POS),
                                                          xyaxes=[0, 1, 0, np.cos(t), 0, np.sin(t)], fovy=WRIST_FOVY)

class CameraRig:
    """Offscreen renderer wrapper. One mujoco.Renderer per (w,h); reuse it, creating GL contexts is slow."""
    def __init__(self, model, width=640, height=480):
        self.m, self.w, self.h = model, width, height
        self._r = mujoco.Renderer(model, height, width)

    def rgb(self, data, cam):
        self._r.update_scene(data, camera=cam); return self._r.render().copy()

    def depth(self, data, cam):
        """Metres along the optical axis; far plane/background = large value."""
        self._r.enable_depth_rendering(); self._r.update_scene(data, camera=cam)
        out = self._r.render().copy(); self._r.disable_depth_rendering(); return out

    def segmentation(self, data, cam):
        """(H,W,2) int32: [object id, object type]; geom id where type==mjOBJ_GEOM."""
        self._r.enable_segmentation_rendering(); self._r.update_scene(data, camera=cam)
        out = self._r.render().copy(); self._r.disable_segmentation_rendering(); return out

    def close(self): self._r.close()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

def intrinsics(model, cam, width, height):
    fovy = np.radians(model.cam_fovy[model.camera(cam).id]); fy = (height / 2) / np.tan(fovy / 2)
    return dict(fx=fy, fy=fy, cx=width / 2, cy=height / 2)
