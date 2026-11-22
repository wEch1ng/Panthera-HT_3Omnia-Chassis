from train_ppo import Task,np,mj
x=Task()
try:
    obs=x.reset(97001); assert obs.shape==(42,)
    # Hidden state changes cannot change the already acquired sensor observation.
    x.env.d.qpos[:2]+=.4;x.env.d.qpos[16:19]+=.2;mj.mj_forward(x.env.m,x.env.d)
    # mj_forward changes simulated motor feedback: hold this sensor packet fixed.
    packet=x.env.motor_feedback(); saved=x.env.motor_feedback
    before=x.obs(); x.env.d.qpos[:2]+=.3;x.env.d.qpos[16:19]-=.1
    x.env.motor_feedback=lambda:packet
    np.testing.assert_array_equal(before,x.obs())
    assert np.isfinite(obs).all()
    print('PASS: 42 sensor-only inputs; hidden base and ball pose excluded.')
finally:x.close()
