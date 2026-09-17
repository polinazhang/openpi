import sys
from unittest.mock import Mock,patch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.franka_real import franka_interface as fi, robot_communicator as rc
with patch.object(fi,'ZMQCameraSubscriber'),patch.object(fi,'FrankaArmOperator') as operator:
 robot=fi.FrankaInterface()
 assert operator.call_args.kwargs['control_mode']=='absolute_eef_pose_to_delta'
 for g,expected in [(-.2,-1.),(0.,1.),(.8,1.)]:
  action=np.array([.5,.1,.2,0,0,0,2,g])
  robot.send_action(action)
  call=robot._operator.arm_control.call_args
  assert not call.args
  np.testing.assert_allclose(call.kwargs['target_pose'],[.5,.1,.2,0,0,0,1])
  assert call.kwargs['gripper_cmd']==expected
 for bad in [np.zeros(8),np.ones(7),np.array([1,1,1,0,0,0,1,np.nan])]:
  robot._operator.arm_control.reset_mock()
  try: robot.send_action(bad)
  except ValueError: pass
  else: raise AssertionError('Bad command accepted')
  robot._operator.arm_control.assert_not_called()
 chunk=np.zeros((50,32));chunk[:,0]=.5;chunk[:,6]=1;chunk[:,7]=-1;chunk[:,8:]=123
 q=rc._action_queue_from_response({'actions':chunk},50)
 assert len(q)==50 and all(a.shape==(8,) for a in q)
 assert rc._config.ROBOT_RUNTIME.max_hz==20
 assert rc._config.ROBOT_RUNTIME.action_horizon==50
print('PASS: configured local imports, mocked constructor, absolute pose keywords, quaternion validation, gripper mapping, 50 x 20 Hz configuration')
