"""Body-frame kiwi drive. x forward, y left, yaw counterclockwise; wheel output rad/s."""
import numpy as np

class OmniDrive:
    def __init__(self, radius=.0654):
        self.radius=radius
        # Wheel centers and joint axes from chassis URDF, ordered left/right/back.
        centers=np.array([[.105242,.182294],[.105242,-.182294],[-.2105,0.]])
        axes=np.array([[-.5,-.866025403784],[-.5,.866025403784],[1.,0.]])
        tangent=np.column_stack((axes[:,1],-axes[:,0]))
        self.matrix=np.column_stack((tangent,-tangent[:,0]*centers[:,1]+tangent[:,1]*centers[:,0]))/radius
        self.inverse=np.linalg.inv(self.matrix)

    def wheels(self,twist,limit=.9*82*np.pi/30):
        wheel=self.matrix@np.asarray(twist,dtype=float)
        return wheel/max(1.,np.max(np.abs(wheel))/limit)

    def body_velocity(self,wheel_velocity):
        return self.inverse@np.asarray(wheel_velocity,dtype=float)
