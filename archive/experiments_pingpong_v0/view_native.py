"""Native MuJoCo UI, external force sliders, D405 overlay, keyboard chassis."""
import argparse
import queue
import time
import threading
import json
from collections import defaultdict
import cv2
import glfw
import numpy as np
import mujoco as mj
import mujoco.viewer
from train_ppo import Task, ActorCritic, torch, HERE
from single_arm import SingleArmTask
from chassis_kinematics import OmniDrive
from view_session import InferenceSession
from native_forces import NativeForces


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--single-arm',action='store_true')
    parser.add_argument('--profile-seconds',type=float,default=0)
    parser.add_argument('--profile-output',default='native_profile.json')
    args=parser.parse_args()
    task=SingleArmTask(force_controls=True) if args.single_arm else Task(force_controls=True)
    if args.single_arm:
        policy=None;iteration=task.iteration
    else:
        checkpoint=torch.load(HERE/'ppo_run/last.pt',map_location='cpu',weights_only=False)
        policy=ActorCritic(42,9,hidden=64);policy.load_state_dict(checkpoint['model']);policy.eval()
        iteration=checkpoint['iteration']
    session=InferenceSession(task);env=task.env
    drive=OmniDrive();twist=np.zeros(3);forces=NativeForces(env.m)
    env.external_force_callback=forces.apply
    events=queue.SimpleQueue();native_window=[None]
    paused=False;manual=True;release_to_drive=False;notice='';notice_until=0.

    def key_callback(key):
        # Called on the native render thread, which owns this GLFW context.
        # Retain only its window for polling press/release; all mouse callbacks stay native.
        native_window[0]=glfw.get_current_context()
        events.put(key)

    viewer_threads=[]
    threads_before=set(threading.enumerate())
    try:
        with mj.viewer.launch_passive(env.m,env.d,key_callback=key_callback,
                                     show_left_ui=True,show_right_ui=True) as viewer:
            viewer_threads=[t for t in threading.enumerate() if t not in threads_before
                            and t.name.endswith('(_launch_internal)')]
            with viewer.lock():
                viewer.cam.type=mj.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:]=[.1,0,.25];viewer.cam.distance=1.65
                viewer.cam.azimuth=135;viewer.cam.elevation=-22
                viewer.opt.geomgroup[3]=0
                # Show only the external-force channels in Control. This affects UI
                # visibility only; all nine robot actuators still operate normally.
                viewer.opt.actuatorgroup[:]=0;viewer.opt.actuatorgroup[5]=1
            print(f'Native MuJoCo | arm iteration {iteration} | open right Control for force sliders',flush=True)
            timings=defaultdict(list);profile_start=time.monotonic()
            rate_time=time.monotonic();rate_count=0;control_hz=0.
            while viewer.is_running():
                begin=time.monotonic()
                stamp=begin
                viewer.sync(state_only=True)  # transfer controls/state, not millions of mesh vertices
                timings['sync_in_ms'].append((time.monotonic()-stamp)*1000)
                reset=False
                while not events.empty():
                    key=events.get()
                    if key==glfw.KEY_SPACE:paused=not paused
                    elif key==glfw.KEY_R:reset=True
                    elif key==glfw.KEY_X:twist[:]=0
                    elif key==glfw.KEY_C:
                        forces.clear(env.d)
                        with viewer.lock():viewer.perturb.active=viewer.perturb.active2=0
                    elif key==glfw.KEY_M and not args.single_arm:manual=not manual;twist[:]=0
                    elif key in (glfw.KEY_UP,glfw.KEY_DOWN,glfw.KEY_LEFT,glfw.KEY_RIGHT):
                        axis=0 if key in (glfw.KEY_UP,glfw.KEY_DOWN) else 1
                        sign=1 if key in (glfw.KEY_UP,glfw.KEY_LEFT) else -1
                        forces.pulses.append((axis,sign*.006,6))
                window=native_window[0]
                focused=bool(window) and bool(glfw.get_window_attrib(window,glfw.FOCUSED))
                held=lambda key: focused and glfw.get_key(window,key)==glfw.PRESS
                if reset:
                    forces.clear(env.d);session.restart('Manual reset');twist[:]=0;paused=False
                    release_to_drive=True
                    with viewer.lock():viewer.perturb.active=viewer.perturb.active2=0
                desired=np.array([.12*(held(glfw.KEY_W)-held(glfw.KEY_S)),
                                  .12*(held(glfw.KEY_A)-held(glfw.KEY_D)),
                                  .4*(held(glfw.KEY_Q)-held(glfw.KEY_E))])
                if release_to_drive:
                    if not np.any(desired):release_to_drive=False
                    desired[:]=0
                twist+=np.clip(desired-twist,-np.array([.2,.2,.8])/30,np.array([.2,.2,.8])/30)
                if held(glfw.KEY_X) or paused:twist[:]=0
                with viewer.lock():forces.snapshot(viewer.perturb)
                stamp=time.monotonic()
                if not paused:
                    with torch.no_grad():action=None if policy is None else policy.actor(torch.tensor(session.observation)).numpy()
                    restarted=session.advance(action,wheel_override=drive.wheels(twist) if manual else None)
                    forces.pulses[:]=[(a,f,n-1) for a,f,n in forces.pulses if n>1]
                    if restarted:
                        forces.clear(env.d);twist[:]=0;release_to_drive=True
                        with viewer.lock():
                            viewer.perturb.active=viewer.perturb.active2=0
                            viewer.perturb.select=0
                        notice=session.last_restart_reason;notice_until=time.monotonic()+3
                        print(f'Auto restart -> episode {session.episode}: {notice}',flush=True)
                timings['control_ms'].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                # A single state sync per loop is sufficient; the native render thread
                # redraws independently. No runtime mesh/model edits require a full copy.
                timings['sync_out_ms'].append((time.monotonic()-stamp)*1000)
                stamp=time.monotonic()
                rate_count+=int(not paused)
                elapsed=time.monotonic()-rate_time
                if elapsed>=1.:
                    control_hz=rate_count/elapsed;rate_time=time.monotonic();rate_count=0
                label='Original arm' if args.single_arm else 'PPO'
                lines=[f'{label} {iteration} | Episode {session.episode} | {"Paused" if paused else "Running"}',
                       'WASD: drive | Q/E: turn | Space: pause | R: restart',
                       'Right Control: forces (N) | C: clear forces',
                       f'Native mouse: double-click, Ctrl+drag | Control {control_hz:.1f} Hz']
                if time.monotonic()<notice_until:lines.append('Restarted: '+notice)
                if release_to_drive:lines.append('Release drive keys to continue')
                # Native overlays initialize 2D GL state before native set_images.
                viewer.set_texts((int(mj.mjtFont.mjFONT_NORMAL),int(mj.mjtGridPos.mjGRID_TOPLEFT),'\n'.join(lines),''))
                rect=viewer.viewport
                if rect and rect.width>160 and rect.height>120:
                    iw=4*(min(320,rect.width//3)//4);ih=iw*3//4
                    image=cv2.resize(env.latest_rgb,(iw,ih),interpolation=cv2.INTER_NEAREST)
                    cv2.rectangle(image,(0,0),(iw-1,ih-1),(240,240,240),2)
                    cv2.rectangle(image,(2,2),(iw-3,24),(25,25,25),-1)
                    cv2.putText(image,'D405 RGB - policy input',(7,18),cv2.FONT_HERSHEY_SIMPLEX,.42,(255,255,255),1)
                    viewer.set_images((mj.MjrRect(rect.left+rect.width-iw-12,rect.bottom+12,iw,ih),image))
                timings['overlay_ms'].append((time.monotonic()-stamp)*1000)
                time.sleep(max(0,1/30-time.monotonic()+begin))
                timings['loop_ms'].append((time.monotonic()-begin)*1000)
                if args.profile_seconds and time.monotonic()-profile_start>=args.profile_seconds:
                    report={k:dict(mean=float(np.mean(v)),p95=float(np.percentile(v,95))) for k,v in timings.items()}
                    report['control_hz']=1000/report['loop_ms']['mean']
                    (HERE/args.profile_output).write_text(json.dumps(report,indent=2))
                    print(json.dumps(report),flush=True)
                    break
    finally:
        # launch_passive closes asynchronously. Wait for its GL thread before
        # destroying offscreen rendering or letting GLFW's atexit hook terminate.
        for thread in viewer_threads:thread.join(timeout=5)
        env.external_force_callback=None;forces.clear(env.d);task.close()


if __name__=='__main__':main()
