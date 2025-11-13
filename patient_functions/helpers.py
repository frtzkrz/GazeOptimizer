def convert_metric(metric):
    #weight: 'D2_Macula'
    #returns: {'roi_name': 'Macula', 'metric_type': D, 'value': 2}

    what, roi_name = metric.split('_')
    metric_type = what[0]
    value = int(what[1:])
    return {'roi_name': roi_name, 'metric_type': metric_type , 'value': value}

def convert_metrics(metrics):
    return [convert_metric(metric) for metric in metrics]

def convert_weights(weights):
    new_weights = []
    for weight in weights:
        new_weight = convert_metric(weight)
        new_weight['weight'] = weights[weight]
        new_weights.append(new_weight)
    return new_weights
