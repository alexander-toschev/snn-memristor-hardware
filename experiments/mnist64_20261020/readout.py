"""Fixed readout, fitted before official test labels are opened."""
import json
import numpy as np
from experiments.mnist_accuracy_20261020.analyze import fit_ridge,predict,evaluate
from experiments.cifar_resumable_20261019.durable import atomic_json,sha
from experiments.mnist64_20261020 import inputs


def features(observations):
    rows=observations['outcomes']
    expected=[('readout_train',i) for i in range(128)]+[('official_test',100000+i) for i in range(100)]
    if [(r['phase'],r['train_index']) for r in rows]!=expected:raise ValueError('Readout split/order differs')
    return np.asarray([[s for n in r['neurons'] for s in n['posts']] for r in rows],dtype=float)


def classify(observations,output,progress=lambda s:None):
    x=features(observations)
    if x.shape!=(228,128):raise ValueError('Expected4x32 raster')
    progress('Readout fit: 128 frozen train responses, ridge lambda=1')
    model=fit_ridge(x[:128],inputs.labels('train'))
    atomic_json(output/'READOUT.json',model)
    # Test labels are first opened only AFTER the model bytes have been saved.
    test_labels=inputs.labels('test')
    result=evaluate(predict(model,x[128:]),x[128:],test_labels)
    result.update(schema='mnist64-frozen-readout-v1',readout_sha256=sha(output/'READOUT.json'),
        train_spikes=int(x[:128].sum()),test_spikes=int(x[128:].sum()),separate_control_bank=False,
        train_label_sha256=sha(inputs.HERE/'labels_train.json'),test_label_sha256=sha(inputs.HERE/'labels_test.json'),
        note='Pilot100 fixed official test prefix; silent/tied abstain counted incorrect; not full10000 accuracy')
    np.save(output/'features.npy',x,allow_pickle=False);atomic_json(output/'ACCURACY.json',result)
    progress('Accuracy: strict=%d/100; argmax=%d/100; coverage=%.2f'%(result['strict_correct'],result['conventional_argmax_correct'],result['coverage']))
    return result
