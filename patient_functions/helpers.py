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

# Print iterations progress
def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix} ({iteration}/{total})', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()