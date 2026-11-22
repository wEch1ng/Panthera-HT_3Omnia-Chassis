"""Ray picking + mouse perturbation through the actual 16-substep plant loop."""
import numpy as np
import mujoco as mj
from balance import Env, HOME
from mouse_force import MouseForce


def main():
    env=Env(); renderer=None
    try:
        env.reset(81000)
        camera=mj.MjvCamera();camera.lookat[:]=[.1,0,.25]
        camera.distance=1.65;camera.azimuth=135;camera.elevation=-22
        opt=mj.MjvOption();opt.geomgroup[3]=0
        renderer=mj.Renderer(env.m,height=480,width=640)
        renderer.enable_segmentation_rendering()
        renderer.update_scene(env.d,camera=camera,scene_option=opt)
        segmentation=renderer.render()
        mouse=MouseForce(env.m,env.d,renderer.scene,opt)
        # Pick visible pixels of ball, arm and chassis; the same ray path used by GLFW.
        for name in ('ball','link3','base_link'):
            body=env.m.body(name).id
            geoms=np.flatnonzero(env.m.geom_bodyid==body)
            ys,xs=np.where(np.isin(segmentation[:,:,0],geoms)&(segmentation[:,:,1]==mj.mjtObj.mjOBJ_GEOM))
            assert len(xs),f'{name} not visible in validation camera'
            middle=np.argmin((xs-xs.mean())**2+(ys-ys.mean())**2)
            assert mouse.select((xs[middle]+.5)/640,1-(ys[middle]+.5)/480,640/480)
            assert mouse.perturb.select==body,(name,mouse.selected_name)
            assert mouse.begin(button=1)
            before=mouse.perturb.refpos.copy()
            mouse.move(.012,0,horizontal=True)
            assert np.linalg.norm(mouse.perturb.refpos-before)>0
            mouse.apply_forces(env.m,env.d)
            assert np.linalg.norm(env.d.xfrc_applied[body,:3])>0
            print('Picked and dragged',name,mouse.last_wrench,flush=True)
            mouse.release();np.testing.assert_array_equal(env.d.xfrc_applied,0)

        # Force reaches dynamics, at every substep, without editing joint positions.
        def rollout(drag):
            env.reset(81000)
            renderer.update_scene(env.d,camera=camera,scene_option=opt)
            mouse.reset();mouse.perturb.select=env.m.body('ball').id
            mouse.begin();mouse.move(.02,0,horizontal=True)
            calls=[]
            def update(m,d):
                calls.append(float(d.time))
                if drag:mouse.apply_forces(m,d)
                else:d.xfrc_applied[:]=0
            env.external_force_callback=update
            env.apply_command(HOME,np.zeros(3))
            assert len(calls)==16 and all(b>a for a,b in zip(calls,calls[1:]))
            return env.d.qvel[15:18].copy()
        base=rollout(False);pushed=rollout(True)
        assert np.linalg.norm(pushed-base)>.001,(base,pushed)
        mouse.release();env.external_force_callback=lambda m,d:mouse.apply_forces(m,d)
        env.apply_command(HOME,np.zeros(3));np.testing.assert_array_equal(env.d.xfrc_applied,0)

        # Rotation and keyboard pulse combine; reset cancels the interaction.
        mouse.perturb.select=env.m.body('base_link').id
        mouse.begin(rotate=True);mouse.move(.01,0,horizontal=True)
        mouse.apply_forces(env.m,env.d,[('ball',0,.006,6)])
        assert np.linalg.norm(mouse.last_wrench[3:])>0
        assert env.d.xfrc_applied[env.m.body('ball').id,0]==.006
        mouse.reset();assert mouse.perturb.select==0 and not mouse.perturb.active
        np.testing.assert_array_equal(env.d.xfrc_applied,0)
        print('PASS: ball/arm/base picking, force/torque, 16 substeps, additive pulses, release/reset.')
    finally:
        env.external_force_callback=None
        if renderer is not None:renderer.close()
        env.renderer.close()


if __name__=='__main__':main()
