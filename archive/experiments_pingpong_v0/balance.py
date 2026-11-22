"""Train, evaluate or view a structured whole-body ball-centering policy."""
import argparse, json, time
from collections import deque
from perception.rgbd_ball import locate_ball
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from motor_limits import MotorFeedback, NAMES as MOTOR_NAMES, STALL_TORQUE, RATED_SPEED, SAFETY, clip_command

HERE = Path(__file__).resolve().parent
HOME = np.array([0., 1., 1.7, -.7, 0., 0.])
NAMES = ['left_wheel', 'right_wheel', 'back_wheel']
R = .0654


def model(force_controls=False):
    src = HERE.parent/'mujoco'
    root = ET.parse(src/'scene.xml').getroot()
    root.remove(root.find('keyframe'))
    w = root.find('worldbody'); assets = root.find('asset')
    for mesh in assets.findall('mesh'):
        mesh.set('file', str(src/mesh.get('file')))
    # Replace the entire arm with the converted reference URDF, including camera_link.
    arm_dir = HERE/'reference_arm/converted'
    reference = ET.parse(arm_dir/'scene.xml').getroot()
    arm = reference.find("worldbody/body[@name='base_link']")
    chassis = w.find("body[@name='base_link']")
    previous = chassis.find("body[@name='arm_base_link']")
    arm.set('name','arm_base_link')
    for attr in ('pos','quat'):
        if previous.get(attr): arm.set(attr,previous.get(attr))
    chassis.remove(previous); chassis.append(arm)
    for mesh in reference.findall('asset/mesh'):
        original_name=mesh.get('name'); mesh.set('name','ref_'+original_name)
        mesh.set('file',str(arm_dir/mesh.get('file'))); assets.append(mesh)
    for geom in arm.iter('geom'):
        if geom.get('mesh'): geom.set('mesh','ref_'+geom.get('mesh'))
    paddle = ET.parse(HERE/'paddle/paddle.xml').getroot()
    for item in paddle.find('asset'):
        if item.tag == 'mesh': item.set('file', str(HERE/'paddle'/item.get('file')))
        assets.append(item)
    grip = w.find(".//body[@name='gripper_center_link']")
    for name, y in [('L', .014789), ('R', -.015211)]:
        finger = grip.find(f"body[@name='gripper_{name}_link']"); pos = finger.get('pos').split(); pos[1] = str(y); finger.set('pos',' '.join(pos))
    wrist = arm.find(".//body[@name='camera_link']")
    # URDF camera_link is +Z forward/+Y down; MuJoCo camera is -Z forward/+Y up.
    ET.SubElement(wrist,'camera',name='wrist',pos='0 0 0',quat='0 1 0 0',fovy='58')
    tool = ET.SubElement(grip,'body',pos='-0.098 0 0',quat='-0.5 -0.5 0.5 0.5'); blade = paddle.find(".//body[@name='paddle']"); tool.append(blade)
    for g in blade.findall('geom'):
        g.attrib.pop('class', None); coll = g.get('name') == 'paddle_blade'; g.set('contype','4' if coll else '0'); g.set('conaffinity','8' if coll else '0')
    ball = paddle.find(".//body[@name='ball']"); w.append(ball); ball.find('geom').attrib.update(contype='8',conaffinity='5'); w.find("geom[@name='floor']").set('conaffinity','10')
    # Isotropic vertical support; longitudinal traction applied at the loaded contact only.
    for name in NAMES:
        b = w.find(f".//body[@name='{name}']"); b.find('joint').set('damping','.01'); g = b.find("geom[@group='3']"); g.attrib.clear(); g.attrib.update(name=name+'_contact', type='sphere',size=str(R),group='3',contype='2',conaffinity='1',condim='1',priority='2')
    for name,stall in zip(MOTOR_NAMES,STALL_TORQUE):
        w.find(f".//joint[@name='{name}']").set('actuatorfrcrange',f'{-stall} {stall}')
        root.find(f"actuator/motor[@joint='{name}']").set('ctrlrange',f'{-stall} {stall}')
    root.find('option').set('timestep',str(1/480))
    if force_controls:
        from native_forces import add_force_controls
        add_force_controls(root)
    return mj.MjModel.from_xml_string(ET.tostring(root,encoding='unicode'))


class SensorPolicy:
    """Controller boundary: fixed calibration, RGB-D detection and wheel encoders only."""
    def __init__(self,origin,basis,home_R,ik,axes):
        self.origin,self.basis,self.home_R,self.ik,self.axes=origin,basis,home_R,ik,axes

    def reset(self,wheel_positions):
        self.prev=None; self.vel=np.zeros(2); self.wheel_ref=wheel_positions.copy(); self.delay=deque([(HOME.copy(),np.zeros(3)) for _ in range(2)]); self.visible=0; self.frames=0; self.missing=0

    def command(self,point,wheel_positions,gains):
        self.frames+=1; kp,kd,wp,wd=gains
        if point is not None:
            self.visible+=1; err_local=self.basis.T@(point-self.origin); err_local[2]=0; e=(self.home_R@err_local)[:2]
            raw=np.zeros(2) if self.prev is None else (e-self.prev)*30/(1+self.missing); self.vel=.55*self.vel+.45*raw; self.prev=e; self.missing=0
        else:
            self.missing+=1; e=np.zeros(2) if self.prev is None or self.missing>3 else self.prev+self.vel*self.missing/30
            if self.missing>3: self.vel[:]=0; self.prev=None
        tilt=np.clip(kp*e+kd*self.vel,-.13,.13); target=HOME+self.ik@[tilt[1],-tilt[0]]
        tangents=np.column_stack((self.axes[:,1],-self.axes[:,0])); trips=(wheel_positions-self.wheel_ref)*R; odom=(2/3)*(tangents.T@trips); speed=np.clip(wp*e+wd*self.vel-.6*odom,-.18,.18); omega=tangents@speed/R
        self.delay.append((target,omega)); target,omega=self.delay.popleft()
        return target,omega


class Env:
    def __init__(self,force_controls=False):
        self.m = m = model(force_controls=force_controls); self.d = mj.MjData(m); self.blade=m.geom('paddle_blade').id; self.ball=m.geom('ball_geom').id
        self.external_force_callback=None
        self.wheels={m.geom(n+'_contact').id:(m.body(n).id,m.joint(n+'_joint').id) for n in NAMES}; self.floor=m.geom('floor').id
        self.force=np.zeros(6); self.zeros=np.zeros(3); self.jac=np.zeros((3,m.nv)); self.jacr=self.jac.copy()

        self.renderer=mj.Renderer(m,height=96,width=128); self.opt=mj.MjvOption(); self.opt.geomgroup[3]=0
        self.kin=mj.MjData(m); self.kin.qpos[10:16]=HOME; mj.mj_forward(m,self.kin)
        k=self.kin; cam=m.camera('wrist').id; self.cam=cam; self.camgeom = np.array([g for g in range(m.ngeom) if m.geom_type[g]==mj.mjtGeom.mjGEOM_MESH and m.geom_group[g]!=3]); self.corners=np.array([[x,y,z] for x in [-1,1] for y in [-1,1] for z in [-1,1]]); self.box=m.geom_aabb[self.camgeom,None,:3]+self.corners*m.geom_aabb[self.camgeom,None,3:]
        # Rigid calibration, expressed in wrist camera coordinates, computed from CAD only.
        self.origin=k.cam_xmat[cam].reshape(3,3).T@(k.geom_xpos[self.blade]-k.cam_xpos[cam]); self.basis=k.cam_xmat[cam].reshape(3,3).T@k.geom_xmat[self.blade].reshape(3,3)
        mj.mj_jacGeom(m,k,self.jac,self.jacr,self.blade); self.ik=np.linalg.pinv(self.jacr[:2,9:15]); self.axes=k.xaxis[1:4].copy(); self.home_R=k.geom_xmat[self.blade].reshape(3,3).copy()
        self.actor=SensorPolicy(self.origin,self.basis,self.home_R,self.ik,self.axes)
        self.gravity=np.array(k.qfrc_bias[9:15]) # nominal feedforward from CAD at HOME, independent of plant

    def motor_feedback(self):
        # One-DOF, gear=1 actuators: applied motor torque, never contact or bias force.
        return MotorFeedback(self.d.qpos[7:16].copy(),self.d.qvel[6:15].copy(),self.d.actuator_force[:9].copy())

    def observe(self):
        r=self.renderer; r.update_scene(self.d,camera='wrist',scene_option=self.opt); r.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=0
        # Cull only CAD meshes whose entire bounding box is outside one frustum plane.
        # Offscreen geometry cannot occlude any RGB or depth pixel (shadows/reflectance off).
        d=self.d; idx=self.camgeom; corners=(self.box@d.geom_xmat[idx].reshape(-1,3,3).transpose(0,2,1)+d.geom_xpos[idx,None,:]-d.cam_xpos[self.cam])@d.cam_xmat[self.cam].reshape(3,3); x,y,z=corners.transpose(2,0,1); hy=-z*.56+.002; hx=-z*.75+.002
        out=set(idx[(x>hx).all(1)|(x<-hx).all(1)|(y>hy).all(1)|(y<-hy).all(1)|(z>0).all(1)])
        for geom in r.scene.geoms[:r.scene.ngeom]:
            if geom.objtype==mj.mjtObj.mjOBJ_GEOM and geom.objid in out: geom.type=mj.mjtGeom.mjGEOM_SPHERE; geom.size[:]=0
        r.scene.flags[mj.mjtRndFlag.mjRND_REFLECTION]=0
        rgb=r.render(); self.latest_rgb=rgb.copy(); r.enable_depth_rendering(); depth=r.render(); r.disable_depth_rendering(); local,_=locate_ball(rgb,depth,58)
        return local

    def reset(self, seed):
        m,d=self.m,self.d; rng=np.random.default_rng(seed); mj.mj_resetData(m,d); d.qpos[10:16]=HOME; mj.mj_forward(m,d)
        self.center=d.geom_xpos[self.blade].copy(); normal=d.geom_xmat[self.blade].reshape(3,3)[:,2]; assert normal[2]>.999
        a,b=rng.uniform(0,2*np.pi,2); offset=rng.uniform(.012,.045)*np.array([np.cos(a),np.sin(a)]); d.qpos[16:19]=self.center+[ *offset,.0281]; d.qvel[15:17]=rng.uniform(.015,.09)*np.array([np.cos(b),np.sin(b)]); mj.mj_forward(m,d); self.actor.reset(d.qpos[7:10]); self.wdist=0.; self.lastbase=d.qpos[:2].copy()

    def error(self):
        v=self.d.geom_xpos[self.ball]-self.d.geom_xpos[self.blade]; rot=self.d.geom_xmat[self.blade].reshape(3,3); return (v-rot[:,2]*np.dot(v,rot[:,2]))[:2]

    def advance(self, gains):
        m,d=self.m,self.d; self.last_motor_observation=self.motor_feedback().observation(); target,omega=self.actor.command(self.observe(),self.motor_feedback().position[:3],gains)
        self.apply_command(target,omega)

    def apply_command(self,target,omega):
        m,d=self.m,self.d
        for _ in range(16):
            if self.external_force_callback is not None:
                self.external_force_callback(m,d)
            feedback=self.motor_feedback()
            requested=np.r_[.6*(omega-feedback.velocity[:3]),
                self.gravity+(target-feedback.position[3:])*[240,360,300,180,100,80]
                -feedback.velocity[3:]*[18,24,20,12,8,6]]
            d.ctrl[:9]=clip_command(requested,feedback.velocity)
            d.qfrc_applied[:]=0
            for i,c in enumerate(d.contact):
                g1,g2=c.geom; wheel=self.wheels.get(g2 if g1==self.floor else g1 if g2==self.floor else -1)
                if wheel is None: continue
                body,j=wheel; axis=d.xaxis[j]; t=np.array([axis[1],-axis[0],0.]); mj.mj_jac(m,d,self.jac,None,c.pos,body); row=t@self.jac; mj.mj_contactForce(m,d,i,self.force); limit=.8*abs(self.force[0]); d.qfrc_applied[:]+=row*np.clip(-110*(row@d.qvel),-limit,limit)
            mj.mj_step(m,d)
    def training_loss(self,g,seed):
        self.reset(seed); errors=[]
        for _ in range(90):
            self.advance(g); errors.append(85. if self.actor.missing else min(85.,1000*np.linalg.norm(self.actor.prev)))
        return np.mean(errors)+max(errors[-30:])+85*(1-self.actor.visible/self.actor.frames)

    def episode(self,g,seed,sec):
        self.reset(seed); err=[]; held=[]
        for _ in range(round(sec*30)):
            self.advance(g); self.wdist+=np.linalg.norm(self.d.qpos[:2]-self.lastbase); self.lastbase[:]=self.d.qpos[:2]; err.append(np.linalg.norm(self.error())); held.append(any(set(c.geom)=={self.ball,self.blade} for c in self.d.contact));
            if err[-1]>.085 or self.d.geom_xpos[self.ball,2]<self.d.geom_xpos[self.blade,2]-.02: break
        ok=len(err)==round(sec*30) and max(err[-30:])<.02 and all(held[-30:]); lost=sec-len(err)/30
        return dict(visible_fraction=self.actor.visible/self.actor.frames,seed=seed,success=bool(ok),duration_s=len(err)/30,mean_mm=float(np.mean(err)*1000),last_second_max_mm=float(max(err[-30:])*1000),base_travel_mm=self.wdist*1000,loss=float(100*lost+np.mean(err)*1000+max(err[-30:])*1000))


def eval_group(task):
    gains,seeds=task; env=Env()
    try: return [env.episode(gains,s,20) for s in seeds]
    finally: env.renderer.close()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['train','eval','view']); args=ap.parse_args(); path=HERE/'d405_policy.json'; start=np.array([.55,.29,.5,.08]); rng=np.random.default_rng(45)
    if args.mode=='train':
        env=Env(); best=start; score=float('inf'); history=[]
        for i in range(8):
            g=start if i==0 else np.clip(best*np.exp(rng.normal(0,.40,4)),[.2,.08,.1,.05],[5,2,5,2]); loss=np.mean([env.training_loss(g,s) for s in (17,21,32)]); history.append(dict(gains=g.tolist(),loss=loss));
            if loss<score: score,best=loss,g.copy(); path.write_text(json.dumps(dict(gains=best.tolist(),training_loss=score),indent=2))
            print(i,round(loss,2),flush=True)
        (HERE/'d405_training.json').write_text(json.dumps(history,indent=2)); env.renderer.close(); return
    gains=json.loads(path.read_text())['gains'];
    if args.mode=='eval':
        from multiprocessing import get_context
        with get_context('spawn').Pool(4) as pool: rows=sum(pool.map(eval_group,[(gains,range(s,s+5)) for s in range(1000,1020,5)]),[])
        (HERE/'d405_evaluation.json').write_text(json.dumps(rows,indent=2)); print(sum(r['success'] for r in rows),len(rows)); return
    import mujoco.viewer as viewer
    env=Env(); env.reset(1000)
    import cv2
    try:
        with viewer.launch_passive(env.m,env.d,show_left_ui=False,show_right_ui=False) as v:
            with v.lock():
                v.cam.type=mj.mjtCamera.mjCAMERA_FREE
                v.cam.lookat[:]=[.1,0,.25]
                v.cam.distance=1.65
                v.cam.azimuth=135
                v.cam.elevation=-22
                v.opt.geomgroup[3]=0
            while v.is_running():
                begin=time.monotonic()
                env.advance(gains)
                v.sync()
                # Display exactly the RGB frame used by perception, enlarged only here.
                viewport=v.viewport
                if viewport is not None and viewport.width>80 and viewport.height>80:
                    width=min(320,viewport.width//3)
                    height=width*3//4
                    image=cv2.resize(env.latest_rgb,(width,height),interpolation=cv2.INTER_NEAREST)
                    cv2.rectangle(image,(0,0),(width-1,height-1),(225,225,225),2)
                    cv2.rectangle(image,(2,2),(width-3,22),(24,24,24),-1)
                    cv2.putText(image,'D405 RGB',(9,17),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
                    v.set_images((mj.MjrRect(viewport.left+viewport.width-width-12,viewport.bottom+12,width,height),image))
                time.sleep(max(0,1/30-time.monotonic()+begin))
    finally:
        env.renderer.close()

if __name__=='__main__': main()
