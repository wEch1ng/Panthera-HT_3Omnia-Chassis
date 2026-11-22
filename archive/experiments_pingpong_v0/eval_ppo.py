"""Held-out paired comparison; simulator truth is confined to this evaluator."""
import json
from multiprocessing import get_context
from train_ppo import Task,ActorCritic,torch,np,HERE

def evaluate(job):
    seed,mode=job;task=Task();obs=task.reset(seed);ac=ActorCritic(42,9,hidden=64)
    if mode!='baseline':ac.load_state_dict(torch.load(HERE/'ppo_run'/f'{mode}.pt',weights_only=False)['model']);ac.eval()
    errors=[];speeds=[];q=[];dropped=False
    try:
        for t in range(600):
            with torch.no_grad(): action=np.zeros(9) if mode=='baseline' else ac.actor(torch.tensor(obs)).numpy()
            obs,r,done,fail=task.step(action);e=task.env;errors.append(float(np.linalg.norm(e.error())));q.append(e.motor_feedback().position[3:]);speeds.append(float(np.max(np.abs(e.motor_feedback().velocity))))
            dropped=errors[-1]>.085 or e.d.geom_xpos[e.ball,2]<e.d.geom_xpos[e.blade,2]-.02
            if fail or dropped:break
        row=dict(seed=seed,mode=mode,seconds=len(errors)/30,success=bool(len(errors)==600 and max(errors[-30:])<.02 and not dropped),mean_mm=float(np.mean(errors)*1000),final_max_mm=max(errors[-30:])*1000,joint_range_deg=np.rad2deg(np.ptp(q,axis=0)).tolist())
        return row
    finally:task.close()

if __name__=='__main__':
    with get_context('spawn').Pool(4) as pool: rows=pool.map(evaluate,[(seed,mode) for seed in range(81000,81006) for mode in ['baseline','last']])
    (HERE/'ppo_run/evaluation.json').write_text(json.dumps(rows,indent=2));print(json.dumps({mode:sum(r['success'] for r in rows if r['mode']==mode) for mode in ['baseline','last']}))
