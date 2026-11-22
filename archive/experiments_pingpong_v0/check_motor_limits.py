from motor_limits import *
np.testing.assert_allclose(torque_cap(np.zeros(9))[:3],11.7)
np.testing.assert_allclose(torque_cap(SAFETY*NO_LOAD_SPEED),0,atol=1e-12)
for direction in [-1,1]:
    velocity=direction*SAFETY*RATED_SPEED*1.01
    assert np.all(clip_command(direction*np.ones(9)*100,velocity)==0)
    braking=clip_command(-direction*np.ones(9)*100,velocity)
    assert np.all(braking*velocity<0)
    assert np.all(np.abs(braking)<=torque_cap(velocity))
print('PASS: output torque envelope, speed guard, braking in both directions')
