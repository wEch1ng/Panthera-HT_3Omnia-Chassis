"""A blank RGB-D stream must not reveal hidden ball or floating-base motion."""
from balance import Env, np, mj
from perception.rgbd_ball import locate_ball


def main():
    env=Env()
    try:
        blank,_=locate_ball(np.zeros((96,128,3),np.uint8),np.ones((96,128)),58)
        assert blank is None
        outputs=[]
        for hidden_shift in [0.,.15]:
            env.reset(123); point=env.observe(); assert point is not None
            env.d.qpos[0]+=hidden_shift; env.d.qpos[16]-=hidden_shift; env.d.qvel[15]+=hidden_shift; mj.mj_forward(env.m,env.d)
            sequence=[]
            for frame in [point,point]+[blank]*8:
                q,w=env.actor.command(frame,env.d.qpos[7:10].copy(),[.5,.3,.5,.08]); sequence.append(np.r_[q,w])
            assert env.actor.prev is None; np.testing.assert_array_equal(sequence[-1][6:],np.zeros(3)); outputs.append(sequence)
        np.testing.assert_array_equal(*outputs); print('PASS: same sensor stream, different hidden state, identical commands; blackout cleared stale ball feedback.')
    finally: env.renderer.close()

if __name__=='__main__': main()
