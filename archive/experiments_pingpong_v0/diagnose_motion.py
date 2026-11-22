"""Report joint movement and independently audit long-horizon ball retention."""
from balance import *

def main():
    env=Env(); gains=json.loads((HERE/'d405_policy.json').read_text())['gains']; env.reset(1000)
    joints=[]; errors=[]; lost=None
    try:
        for i in range(1800):
            env.advance(gains); joints.append(env.d.qpos[10:16].copy()); errors.append(float(np.linalg.norm(env.error())))
            if errors[-1]>.085 or env.d.geom_xpos[env.ball,2]<env.d.geom_xpos[env.blade,2]-.02:
                lost=(i+1)/30;break
        result=dict(duration_s=len(errors)/30,drop_time_s=lost,joint_peak_to_peak_deg=np.rad2deg(np.ptp(joints,axis=0)).tolist(),max_error_mm=max(errors)*1000,visible_fraction=env.actor.visible/env.actor.frames,controller='four_gain_visual_feedback_not_PPO',model='migrated_reference_URDF',seed=1000)
        (HERE/'motion_diagnostic.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
    finally: env.renderer.close()

if __name__=='__main__':main()
