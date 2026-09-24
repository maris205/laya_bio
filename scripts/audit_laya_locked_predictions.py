"""Independently check saved test metrics with sklearn/SciPy; never runs a model."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, log_loss, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/laya_locked_test'


def main():
    results = json.loads((OUT/'results.json').read_text())
    assert results['n_models'] == 12
    audit = {'all_passed':False, 'method':'sklearn classification metrics and log loss; SciPy softmax; direct Brier/ECE',
             'new_model_evaluation':False, 'runs':[]}
    membership = None
    for summary in results['runs']:
        name = f"{summary['condition']}_seed{summary['seed']}"
        path = OUT/name/'predictions.jsonl'
        records = [json.loads(x) for x in path.read_text().splitlines()]
        keys = [(r['id'],r['task'],r['label']) for r in records]
        assert len(keys) == 4097 and len({x[0] for x in keys}) == 4097
        if membership is None: membership = keys
        assert keys == membership
        entry = {'name':name, 'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest(), 'tasks':{}}
        for task in results['counts']:
            chosen = [r for r in records if r['task'] == task]
            k = chosen[0]['n_classes']
            y = np.array([r['label'] for r in chosen])
            z = np.array([r['logits'] for r in chosen], dtype=np.float64)
            pred = z.argmax(1)
            assert len(y) == results['counts'][task] and set(y) == set(range(k))
            errors = []
            for mode, temperature in [('raw',1.),('calibrated',summary['temperatures'][task])]:
                p = softmax(z/temperature, axis=1)
                confidence = p.max(1)
                ece = 0.
                for b in range(15):
                    mask = (confidence >= b/15) & (confidence < (b+1)/15 if b<14 else confidence <= 1)
                    if mask.any(): ece += mask.mean()*abs((pred[mask] == y[mask]).mean()-confidence[mask].mean())
                checks = {'accuracy':accuracy_score(y,pred), 'balanced_accuracy':balanced_accuracy_score(y,pred),
                          'macro_f1':f1_score(y,pred,labels=range(k),average='macro',zero_division=0),
                          'mcc':matthews_corrcoef(y,pred), 'nll':log_loss(y,p,labels=range(k)),
                          'brier':np.square(p-np.eye(k)[y]).sum(1).mean(), 'ece15':ece}
                for key, actual in checks.items():
                    error = abs(actual-summary['metrics'][mode][task][key])
                    assert error < 1e-9, (name,task,mode,key,error)
                    errors.append(error)
                if mode == 'raw':
                    assert np.max(np.abs(p-np.array([r['probs'] for r in chosen]))) < 1e-6
            entry['tasks'][task] = {'maximum_metric_error':max(errors),
                                     'true_counts':dict(Counter(map(int,y))),
                                     'predicted_counts':dict(Counter(map(int,pred))),
                                     'confusion_matrix':confusion_matrix(y,pred,labels=range(k)).tolist()}
        audit['runs'].append(entry)
    audit['all_passed'] = True
    (OUT/'independent_metric_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({'all_passed':True,'n_models':len(audit['runs']),
                      'max_error':max(t['maximum_metric_error'] for r in audit['runs'] for t in r['tasks'].values())}))


if __name__ == '__main__': main()
