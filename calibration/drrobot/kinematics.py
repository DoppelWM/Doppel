import mujoco
import pytorch_kinematics.transforms as tf
from pytorch_kinematics import chain, frame
from pytorch_kinematics.mjcf import _build_chain_recurse


def build_chain_from_mjcf_path(path, body=None):
    """pytorch_kinematics Chain from an MJCF file (root = body 0 unless ``body`` is given)."""
    m = mujoco.MjModel.from_xml_path(path)
    root_body = m.body(0) if body is None else m.body(body)
    root_frame = frame.Frame(root_body.name,
                             link=frame.Link(root_body.name, offset=tf.Transform3d(rot=root_body.quat, pos=root_body.pos)),
                             joint=frame.Joint())
    _build_chain_recurse(m, root_frame, root_body)
    return chain.Chain(root_frame)
