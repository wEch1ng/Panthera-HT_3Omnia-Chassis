"""Global interactive view plus the actual D405 policy image."""
import time
import argparse
from chassis_kinematics import OmniDrive
from view_session import InferenceSession
from mouse_force import MouseForce
import glfw
import cv2
from OpenGL import GL
import numpy as np
from train_ppo import Task,ActorCritic,torch,HERE,mj


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--single-arm',action='store_true');args=parser.parse_args()
    if not glfw.init():raise RuntimeError('Cannot initialize desktop window')
    window=glfw.create_window(1280,800,'Panthera PPO | Global + D405 | Space pause | R reset',None,None)
    if not window:raise RuntimeError('Cannot create desktop window')
    glfw.make_context_current(window);glfw.swap_interval(1)
    if args.single_arm:
        from single_arm import SingleArmTask
        task=SingleArmTask();checkpoint={'iteration':task.iteration};policy=None
    else:
        task=Task();checkpoint=torch.load(HERE/'ppo_run/last.pt',weights_only=False)
        policy=ActorCritic(42,9,hidden=64);policy.load_state_dict(checkpoint['model']);policy.eval()
    session=InferenceSession(task);e=task.env
    glfw.make_context_current(window)
    context=mj.MjrContext(e.m,mj.mjtFontScale.mjFONTSCALE_150);scene=mj.MjvScene(e.m,maxgeom=10000);opt=mj.MjvOption();opt.geomgroup[3]=0
    camera=mj.MjvCamera();camera.lookat[:]=[.1,0,.25];camera.distance=1.65;camera.azimuth=135;camera.elevation=-22
    drive=OmniDrive(); twist=np.zeros(3)
    pending=[]
    mouse=MouseForce(e.m,e.d,scene,opt)
    e.external_force_callback=lambda m,d: mouse.apply_forces(m,d,pending)
    mj.mjv_updateScene(e.m,e.d,opt,mouse.perturb,camera,mj.mjtCatBit.mjCAT_ALL,scene)
    push_labels={glfw.KEY_UP:('ball',0,1),glfw.KEY_DOWN:('ball',0,-1),glfw.KEY_LEFT:('ball',1,1),glfw.KEY_RIGHT:('ball',1,-1)}
    state={'paused':False,'reset':False,'cursor':None,'push':'No manual push yet','manual':True,'release_to_drive':False,'notice_until':0.,'last_click':(-1.,-100.,-100.)}
    def key(w,k,sc,a,mods):
        if a==glfw.RELEASE and k in (glfw.KEY_LEFT_CONTROL,glfw.KEY_RIGHT_CONTROL):
            mouse.release()
        if a!=glfw.PRESS:return
        if k in push_labels:
            body,axis,sign=push_labels[k]
            strength=(.006 if body=='ball' else 15.)*(3 if mods&glfw.MOD_SHIFT else 1)
            pending.append([body,axis,sign*strength,6])
            state['push']=f'{body}: {sign*strength:+.3f} N, world {"XY"[axis]}, 0.20 s'
        if k==glfw.KEY_M and not args.single_arm:state['manual']=not state['manual'];twist[:]=0
        if k==glfw.KEY_X:twist[:]=0
        if k==glfw.KEY_SPACE:
            state['paused']=not state['paused']
            if state['paused']:mouse.release()
        if k==glfw.KEY_R:state['reset']=True
        if k==glfw.KEY_ESCAPE:glfw.set_window_should_close(w,True)
    def clear_inputs_after_restart():
        mouse.reset();pending.clear();twist[:]=0;e.d.xfrc_applied[:]=0
        state.update(reset=False,paused=False,release_to_drive=True,push='No manual push yet',
                     notice_until=time.monotonic()+3.)

    def on_inset(w,x,y):
        width,height=glfw.get_window_size(w)
        fw,fh=glfw.get_framebuffer_size(w)
        if min(width,height,fw,fh)<=0:return True
        px,py=x*fw/width,fh-y*fh/height
        iw=min(320,fw//3);ih=iw*3//4
        return fw-iw-12<=px<=fw-12 and 12<=py<=12+ih

    def mouse_button(w,button,action,mods):
        x,y=glfw.get_cursor_pos(w);state['cursor']=(x,y)
        if action==glfw.RELEASE:
            if mouse.drag_button==button:mouse.release()
            return
        if action!=glfw.PRESS or on_inset(w,x,y):return
        width,height=glfw.get_window_size(w)
        if min(width,height)<=0:return
        if mods&glfw.MOD_CONTROL and button in (glfw.MOUSE_BUTTON_LEFT,glfw.MOUSE_BUTTON_RIGHT):
            # Click directly on a body, or drag the body selected by double-click.
            mouse.select(x/width,1-y/height,width/height)
            mouse.begin(rotate=button==glfw.MOUSE_BUTTON_LEFT,button=button)
        elif button==glfw.MOUSE_BUTTON_LEFT:
            now=time.monotonic();last,lx,ly=state['last_click']
            if now-last<.35 and np.hypot(x-lx,y-ly)<8:
                mouse.select(x/width,1-y/height,width/height)
                state['last_click']=(-1.,x,y)
            else:state['last_click']=(now,x,y)

    def cursor(w,x,y):
        previous=state['cursor'];state['cursor']=(x,y)
        if previous is None:return
        width,height=glfw.get_window_size(w)
        if height<=0:return
        dx,dy=(x-previous[0])/height,(y-previous[1])/height
        shift=any(glfw.get_key(w,k)==glfw.PRESS for k in (glfw.KEY_LEFT_SHIFT,glfw.KEY_RIGHT_SHIFT))
        ctrl=any(glfw.get_key(w,k)==glfw.PRESS for k in (glfw.KEY_LEFT_CONTROL,glfw.KEY_RIGHT_CONTROL))
        if mouse.perturb.active:
            mouse.move(dx,dy,horizontal=shift)
            return
        if ctrl or on_inset(w,x,y):return
        left=glfw.get_mouse_button(w,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
        right=glfw.get_mouse_button(w,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
        if left or right:mj.mjv_moveCamera(e.m,mj.mjtMouse.mjMOUSE_MOVE_V if right else mj.mjtMouse.mjMOUSE_ROTATE_V,dx,dy,camera)

    def focus(w,focused):
        if not focused:
            mouse.release();pending.clear();twist[:]=0
            state['cursor']=None

    def scroll(w,x,y):mj.mjv_moveCamera(e.m,mj.mjtMouse.mjMOUSE_ZOOM,0,-.05*y,camera)
    glfw.set_key_callback(window,key);glfw.set_cursor_pos_callback(window,cursor)
    glfw.set_mouse_button_callback(window,mouse_button);glfw.set_window_focus_callback(window,focus)
    glfw.set_scroll_callback(window,scroll)
    print(f"Loaded PPO iteration {checkpoint['iteration']}; global view and D405 inset ready",flush=True)
    try:
        while not glfw.window_should_close(window):
            begin=time.monotonic();glfw.poll_events()
            if state['reset']:
                session.restart('Manual reset')
                clear_inputs_after_restart()
            held=lambda k: glfw.get_key(window,k)==glfw.PRESS
            enabled=bool(glfw.get_window_attrib(window,glfw.FOCUSED))
            desired=np.array([.12*(held(glfw.KEY_W)-held(glfw.KEY_S)),.12*(held(glfw.KEY_A)-held(glfw.KEY_D)),.4*(held(glfw.KEY_Q)-held(glfw.KEY_E))]) if enabled else np.zeros(3)
            if state['release_to_drive']:
                if not np.any(desired):state['release_to_drive']=False
                desired[:]=0
            twist+=np.clip(desired-twist,-np.array([.2,.2,.8])/30,np.array([.2,.2,.8])/30)
            if held(glfw.KEY_X) or state['paused']:twist[:]=0
            if not state['paused']:
                e.d.xfrc_applied[:]=0
                with torch.no_grad():action=None if policy is None else policy.actor(torch.tensor(session.observation)).numpy()
                restarted=session.advance(action,wheel_override=drive.wheels(twist) if state['manual'] else None)
                pending[:]=[[b,a,f,n-1] for b,a,f,n in pending if n>1]
                if restarted:
                    clear_inputs_after_restart()
                    print(f'Auto restart -> episode {session.episode}: {session.last_restart_reason}',flush=True)
            glfw.make_context_current(window)
            width,height=glfw.get_framebuffer_size(window);viewport=mj.MjrRect(0,0,width,height)
            mj.mjv_updateScene(e.m,e.d,opt,mouse.perturb,camera,mj.mjtCatBit.mjCAT_ALL,scene);mj.mjr_render(viewport,scene,context)
            iw=min(320,width//3);ih=iw*3//4
            image=cv2.resize(e.latest_rgb,(iw,ih),interpolation=cv2.INTER_NEAREST)
            cv2.rectangle(image,(0,0),(iw-1,ih-1),(240,240,240),2)
            cv2.rectangle(image,(2,2),(iw-3,24),(25,25,25),-1)
            cv2.putText(image,'D405 RGB - policy input',(9,18),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
            GL.glDisable(GL.GL_DEPTH_TEST)
            GL.glDisable(GL.GL_SCISSOR_TEST)
            GL.glWindowPos2i(width-iw-12,12)
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT,1)
            GL.glDrawPixels(iw,ih,GL.GL_RGB,GL.GL_UNSIGNED_BYTE,np.ascontiguousarray(image[::-1]))
            status=('Manual chassis' if state['manual'] else 'PPO chassis')+(' | Paused' if state['paused'] else '')
            policy_label='Original arm recovery' if args.single_arm else 'PPO'
            lines=[f'{policy_label} {checkpoint["iteration"]} | Episode {session.episode} | Auto restart ON',
                   f'{status} | W/S: forward/back | A/D: left/right | Q/E: turn',
                   'Space: pause | R: restart | X: stop | Arrows: push ball | Shift: stronger',
                   'Double-click: select | Ctrl+right drag: force | Ctrl+left drag: torque',
                   'Shift+drag: horizontal plane | Release mouse: remove force',
                   f'Selected: {mouse.selected_name} | Force: {np.linalg.norm(mouse.last_wrench[:3]):.4f} N | Torque: {np.linalg.norm(mouse.last_wrench[3:]):.4f} Nm']
            if not args.single_arm:lines.append('M: manual/PPO chassis')
            if time.monotonic()<state['notice_until']:
                lines.append('Restarted: '+session.last_restart_reason)
            if state['release_to_drive']:lines.append('Release drive keys before driving again')
            lines.append(state['push']+f' | vx={twist[0]:+.2f} vy={twist[1]:+.2f} yaw={twist[2]:+.2f}')
            mj.mjr_overlay(mj.mjtFont.mjFONT_NORMAL,mj.mjtGridPos.mjGRID_TOPLEFT,viewport,
                           '\n'.join(lines),'',context)
            glfw.swap_buffers(window);time.sleep(max(0,1/30-time.monotonic()+begin))
    finally:
        e.external_force_callback=None;mouse.release()
        glfw.make_context_current(window);context.free();task.close();glfw.destroy_window(window);glfw.terminate()

if __name__=='__main__':main()
