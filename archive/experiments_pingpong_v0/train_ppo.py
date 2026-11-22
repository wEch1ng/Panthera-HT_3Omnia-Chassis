"""Sensor-only residual PPO, nine independent actions; shared reference PPO implementation."""
import argparse,json,time,hashlib
from collections import deque
from pathlib import Path
import numpy as np
import torch
from balance import Env,HOME,HERE,mj
from motor_limits import RATED_SPEED,SAFETY
from ppo import ActorCritic,compute_gae,ppo_update

torch.set_num_threads(1)
class Task:
    def __init__(self,force_controls=False):
        self.env=Env(force_controls=force_controls); self.gains=json.loads((HERE/'d405_policy.json').read_text())['gains'];self.episode=0
    def reset(self,seed):
        self.env.reset(seed);self.rng=np.random.default_rng(seed+12);self.t=0;self.action=np.zeros(9);self.queue=deque([np.zeros(9) for _ in range(2)]);self.sample();return self.obs()
    def sample(self):
        e=self.env;point=e.observe()
        # Noise/dropout is applied to the measured RGB-D estimate, not world truth.
        if point is not None:
            point+=self.rng.normal(0,.0007,3)
            if self.rng.random()<.02:point=None
        self.target,self.omega=e.actor.command(point,e.motor_feedback().position[:3],self.gains)
    def obs(self):
        e=self.env;a=e.actor;f=e.motor_feedback();ball=np.r_[np.zeros(2) if a.prev is None else a.prev/.08,a.vel/.5,float(a.missing==0),min(a.missing/6,1)]
        return np.r_[ball,(f.position[:3]-a.wheel_ref)*.1,(f.position[3:]-HOME)/.2,f.velocity/RATED_SPEED,f.torque/np.r_[[3.5]*3,[6,10,10,6,2,2]],self.action].astype(np.float32)
    def step(self,action,wheel_override=None):
        e=self.env;action=np.clip(action,-1,1);self.queue.append(action.copy());applied=self.queue.popleft()
        e.apply_command(self.target+applied[3:]*.025,self.omega+applied[:3]*.8 if wheel_override is None else wheel_override);self.t+=1
        # Hidden disturbance is an environment event only, never an observation.
        if self.t==60:
            angle=self.rng.uniform(0,2*np.pi);e.d.qvel[15:17]+=.04*np.array([np.cos(angle),np.sin(angle)])
        self.sample();a=e.actor;f=e.motor_feedback();distance=.15 if a.prev is None else np.linalg.norm(a.prev)
        reward=1.-min(distance/.04,4.)-.04*np.sum((action-self.action)**2)-.02*np.sum(action**2)-.002*np.sum(np.abs(f.torque*f.velocity))-.3*(a.missing>0)
        speed_fail=bool(np.any(np.abs(f.velocity)>SAFETY*RATED_SPEED*1.01));failure=a.missing>12 or distance>.1 or speed_fail
        if failure:reward-=5
        self.action=action.copy();return self.obs(),reward,failure or self.t>=300, failure
    def close(self):self.env.renderer.close()


def worker(conn):
    task=Task()
    try:
        while True:
            op,arg=conn.recv()
            if op=='reset':conn.send(task.reset(arg))
            elif op=='step':
                obs,r,done,fail=task.step(arg);terminal=obs.copy()
                if done:task.episode+=1;obs=task.reset(50000+task.episode*31+int(task.rng.integers(10000)))
                conn.send((obs,r,done,fail,terminal))
            else:break
    finally:task.close();conn.close()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--iterations',type=int,default=40);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--steps',type=int,default=64);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
    from multiprocessing import get_context
    out=HERE/'ppo_run';out.mkdir(exist_ok=True);torch.manual_seed(73);np.random.seed(73)
    ac=ActorCritic(42,9,hidden=64,init_std=.18);opt=torch.optim.Adam(ac.parameters(),lr=5e-5);start=0
    if args.resume:
        ck=torch.load(out/'last.pt',weights_only=False);ac.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);start=ck['iteration']
    else:torch.save(dict(model=ac.state_dict(),iteration=0),out/'initial.pt')
    manifest=dict(args=vars(args),observation='RGBD relative ball estimate and velocity, visibility/age, 9 motor position/velocity/torque, previous 9 actions',action='3 wheel speed residuals and 6 joint position residuals',algorithm='reference PPO / sensor-only actor and critic / sensor-only reward',dt=1/30,delay_frames=2,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2));ctx=get_context('spawn');pipes=[];processes=[]
    for i in range(args.workers):
        parent,child=ctx.Pipe();p=ctx.Process(target=worker,args=(child,));p.start();child.close();pipes.append(parent);processes.append(p);parent.send(('reset',700+i))
    obs=np.array([p.recv() for p in pipes]);begin=time.time()
    try:
        for iteration in range(start,start+args.iterations):
            buf={k:[] for k in ['obs','act','logp','rew','val','next','done']};failures=0
            for t in range(args.steps):
                with torch.no_grad():act,logp,val=ac.act(torch.tensor(obs))
                for p,a in zip(pipes,act.numpy()):p.send(('step',a))
                results=[p.recv() for p in pipes];nextobs,reward,done,fail,terminal=map(np.array,zip(*results));failures+=sum(fail)
                with torch.no_grad():nv=ac.value(torch.tensor(terminal)).numpy();nv[fail]=0
                for k,v in zip(buf,[obs,act.numpy(),logp.numpy(),reward,val.numpy(),nv,done]):buf[k].append(v)
                obs=nextobs
            b={k:np.array(v) for k,v in buf.items()};adv,ret=compute_gae(b['rew'],b['val'],b['next'],b['done']);batch={k:torch.tensor(v.reshape((-1,)+v.shape[2:]),dtype=torch.float32) for k,v in dict(obs=b['obs'],act=b['act'],logp=b['logp'],adv=adv,ret=ret).items()}
            metrics=ppo_update(ac,opt,batch,epochs=4,minibatches=4,ent_coef=.001);row=dict(iteration=iteration+1,mean_reward=float(b['rew'].mean()),failures=int(failures),elapsed_s=time.time()-begin,**metrics)
            with (out/'log.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            torch.save(dict(model=ac.state_dict(),optimizer=opt.state_dict(),iteration=iteration+1,obs_dim=42,act_dim=9),out/'last.pt');print(json.dumps(row),flush=True)
    finally:
        for p in pipes:p.send(('close',None))
        for p in processes:p.join(20)

if __name__=='__main__':main()
