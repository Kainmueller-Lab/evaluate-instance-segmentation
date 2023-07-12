import numpy as np
import csv
from functools import reduce

def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else
        default, keys.split("."), dictionary
    )

def summarize_metric_dict(metric_dicts, names, metrics, output_name, agg_inst_dict=None):

    csvf = open(output_name, 'w', newline='')
    writer = csv.writer(csvf, delimiter=';',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    print(metrics)
    header = ['sample'] + [m.split('.')[-2] + ' ' + m.split('.')[-1] for m in
                      metrics]
    writer.writerow(header)

    summary = np.zeros((len(metric_dicts), len(metrics)))
    for i, (name, metric_dict) in enumerate(zip(names, metric_dicts)):
        for k in range(len(metrics)):
            v = deep_get(metric_dict, metrics[k])
            if v is not None:
                summary[i, k] = float(v)
            else:
                summary[i, k] = 0.0

        writer.writerow([name] + list(summary[i]))
    writer.writerow(['mean'] + list(np.mean(summary, axis=0)))
    writer.writerow(['sum'] + list(np.sum(summary, axis=0)))
    
    # write average over instances
    if agg_inst_dict is not None:
        avg_inst = []
        for m in metrics:
            v = deep_get(agg_inst_dict, m)
            if v is not None:
                avg_inst.append(v)
            else:
                avg_inst.append(0.0)
        writer.writerow(['avg_inst'] + avg_inst)

    csvf.close()
