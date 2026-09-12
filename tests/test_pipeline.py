import numpy as np
import pytest
import torch
from cgddnet.metrics import binary_metrics, summarize
from cgddnet.inference import sliding_window
from cgddnet.schedule import WarmupCosine


def test_masked_metrics_and_undefined_auc():
    p=np.array([[0.9,0.8],[0.2,0.99]])
    y=np.array([[1,0],[1,0]])
    mask=np.array([[1,1],[1,0]])
    result=binary_metrics(p,y,mask)
    assert (result['TP'],result['TN'],result['FP'],result['FN'])==(1,0,1,1)
    assert result['F1']==pytest.approx(.5)
    one_class=binary_metrics(np.ones((2,2))*.1,np.zeros((2,2)),np.ones((2,2)))
    assert one_class['AUC'] is None
    stats=summarize([result,one_class])
    assert stats['metrics']['AUC']['defined_images']==1
    assert stats['metrics']['AUC']['undefined_images']==1
    with pytest.raises(ValueError,match='Empty'): binary_metrics(p,y,np.zeros((2,2)))


def test_sliding_window_keeps_original_geometry_and_averages_probabilities():
    class IdentityLogit(torch.nn.Module):
        def forward(self,x): return x[:,:1]
    image=np.linspace(-2,2,23*37,dtype=np.float32).reshape(1,23,37)
    model=IdentityLogit().train()
    actual=sliding_window(model,image,patch_size=16,stride=7,batch_size=3)
    np.testing.assert_allclose(actual,1/(1+np.exp(-image[0])),atol=2e-7)
    assert actual.shape==(23,37) and model.training
    tiny=np.ones((1,1,5),dtype=np.float32)
    assert sliding_window(model,tiny,16,8).shape==(1,5)
    with pytest.raises(ValueError): sliding_window(model,image,16,17)


def test_warmup_restart_boundaries_and_scheduler_resume():
    schedule=WarmupCosine(4,6,2,.1,.01)
    assert schedule(0)==pytest.approx(.1)
    assert schedule(4)==pytest.approx(1)
    assert schedule(9)<schedule(8)
    assert schedule(10)==pytest.approx(1)
    assert schedule(22)==pytest.approx(1)
    p=torch.nn.Parameter(torch.ones(1)); optim=torch.optim.Adam([p],lr=.01)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optim,schedule)
    for _ in range(9): optim.step(); scheduler.step()
    saved_optim,saved_scheduler=optim.state_dict(),scheduler.state_dict()
    optim2=torch.optim.Adam([torch.nn.Parameter(torch.ones(1))],lr=.01)
    scheduler2=torch.optim.lr_scheduler.LambdaLR(optim2,WarmupCosine(4,6,2,.1,.01))
    optim2.load_state_dict(saved_optim); scheduler2.load_state_dict(saved_scheduler)
    for _ in range(15):
        optim.step(); scheduler.step(); optim2.step(); scheduler2.step()
        assert scheduler.get_last_lr()==scheduler2.get_last_lr()
