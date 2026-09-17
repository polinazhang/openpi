import dataclasses
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.franka_real import checkpoint_policy as cp
from examples.franka_real import config
from openpi.shared import normalize
from openpi import transforms
from openpi.training import config as training
from openpi.policies import policy_config
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)
 try: cp.load_checkpoint_stats(root)
 except FileNotFoundError: pass
 else: raise AssertionError('Missing checkpoint stats did not fail')
 stats={key: normalize.NormStats(mean=np.zeros(8),std=np.ones(8),q01=np.arange(8,dtype=float),q99=np.arange(8,dtype=float)+2) for key in ['state','actions']}
 path=root/'assets/franka/norm_stats.json';path.parent.mkdir(parents=True);path.write_text(normalize.serialize_json(stats))
 checkpoint,_,loaded,_=cp.load_checkpoint_stats(root)
 tc=cp.make_inference_config(checkpoint,loaded)
 assert (tc.model.action_dim,tc.model.action_horizon,tc.model.max_token_len)==(32,50,200)
 assert tc.model.pi05
 with patch.object(training.DataConfigFactory,'_load_norm_stats',side_effect=AssertionError('Fallback attempted')), patch.object(training.ModelTransformFactory,'__call__',return_value=transforms.Group()):
  dc=tc.data.create(Path('/tmp/unrelated-assets'),tc.model)
 assert dc.norm_stats is loaded and dc.use_quantile_norm
 assert not dc.data_transforms.outputs
 x=np.zeros((50,32));x[:,8:]=999
 physical=transforms.Unnormalize(loaded,use_quantiles=True)({'actions':x,'state':np.zeros(8)})
 np.testing.assert_allclose(physical['actions'][0,:8],np.arange(8)+1.0000005)
 with patch.object(policy_config,'create_trained_policy',return_value='fake') as create:
  cfg=dataclasses.replace(config.POLICY_SERVER,checkpoint_dir=str(root),norm_stats_path=str(path))
  assert cp.load_policy(cfg)=='fake'
  assert create.call_args.kwargs['norm_stats'] is not None
  assert create.call_args.args[0].policy_metadata['norm_stats_path']==str(path)
  bad=dataclasses.replace(cfg,norm_stats_path='/tmp/unrelated-assets/norm_stats.json')
  try: cp.load_policy(bad)
  except ValueError: pass
  else: raise AssertionError('Foreign stats accepted')
 path.unlink();outside=root.parent/(root.name+'-external-stats.json');outside.write_text(normalize.serialize_json(stats))
 try:
  path.symlink_to(outside)
  try: cp.load_checkpoint_stats(root)
  except ValueError: pass
  else: raise AssertionError('External stats symlink accepted')
 finally: outside.unlink()
print('PASS: exact checkpoint stats only, missing/foreign stats rejected, quantile transform once, Pi0.5 32/50/200, no asset fallback')
