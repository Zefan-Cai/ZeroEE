import json

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0

def compute_f1(predicted, gold, matched):
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return precision, recall, f1


def cal_scores(gold_triggers, pred_triggers, gold_events, pred_events):
    assert len(gold_triggers) == len(pred_triggers)
    assert len(gold_events) == len(pred_events)  
    # tri_id
    gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_triggers, pred_triggers):
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)

        gold_tri_id_num += len(gold_set)
        pred_tri_id_num += len(pred_set)
        match_tri_id_num += len(gold_set & pred_set)
    
    # tri_cls
    gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
    for gold_trigger, pred_trigger in zip(gold_events, pred_events):
        gold_set = set(gold_trigger)
        pred_set = set(pred_trigger)
        gold_tri_cls_num += len(gold_set)
        pred_tri_cls_num += len(pred_set)
        match_tri_cls_num += len(gold_set & pred_set)
    
    scores = {
        'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
        'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
    }
    
    return scores

def get_trigger(examples):
    all_outputs = []
    test_gold_triggers, test_gold_events, test_pred_triggers, test_pred_events = [], [], [], []


    for example in examples:
        pred_object = []
        gt_trigger_object = []
        gt_event_object = []
        
        for sub_example in example:
            event_type = sub_example["Event type"][0]
            trigger = sub_example["trigger"]
            raw_output = sub_example["prediction"]
            
            if trigger != '<trigger>':
                gt_trigger_object.extend(trigger)
                gt_event_object.append(event_type)
            
            try:
                triggers = raw_output.split('Event trigger is ', 1)[1]
                triggers = triggers.split(' and ')
                for t_cnt, t in enumerate(triggers):
                    if t != '<trigger>':
                        pred_object.append((t, event_type, {'tri counter': t_cnt})) # (text, type, kwargs)
            except:
                pass
            sub_example["trigger"] = pred_object
            all_outputs.append(pred_object)
                
        pred_trigger_object = []
        pred_event_object = []
        
        for obj in pred_object:
            pred_event_object.append(obj[1])
            pred_trigger_object.append(obj[0])

        test_gold_triggers.append(gt_trigger_object)
        test_gold_events.append(gt_event_object)
        test_pred_triggers.append(pred_trigger_object)
        test_pred_events.append(pred_event_object)


    return test_gold_triggers, test_pred_triggers, test_gold_events, test_pred_events



examples = []
with open('/local1/zefan/results/test_Llama-2-7b-geneva-20-96-5-epoch10/predictions/ACE_valid.jsonl', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        examples.append(json.loads(line))


test_gold_triggers, test_pred_triggers, test_gold_events, test_pred_events = get_trigger(examples)
test_scores = cal_scores(test_gold_triggers, test_pred_triggers, test_gold_events, test_pred_events)

print("---------------------------------------------------------------------")
print('Trigger I  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_id'][3] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][1], 
    test_scores['tri_id'][4] * 100.0, test_scores['tri_id'][2], test_scores['tri_id'][0], test_scores['tri_id'][5] * 100.0))
print('Trigger C  - P: {:6.2f} ({:4d}/{:4d}), R: {:6.2f} ({:4d}/{:4d}), F: {:6.2f}'.format(
    test_scores['tri_cls'][3] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][1], 
    test_scores['tri_cls'][4] * 100.0, test_scores['tri_cls'][2], test_scores['tri_cls'][0], test_scores['tri_cls'][5] * 100.0))