
import datasets

import numpy as np

from sklearn.metrics import (
    classification_report
)


# @datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class EE(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description="hello",
            citation="hello",
            inputs_description="hello",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32")
                }
            ),
            codebase_urls=["hello"],
            reference_urls=["hello"]
        )

    def _compute(self, predictions, references):
        
        scores = self._scores(predictions, references)
        
        return scores
    
    def _accuracy(self, predictions, references):
        Correct = 0
        for pred, gt in zip(predictions, references):
            if pred == gt:
                Correct += 1 
        return Correct/len(predictions)
    
    def _scores(self, predictions, references):
        
        total_labels = []
        total_prelabels = []
        
        Correct = 0
        
        for pred, gt in zip(predictions, references):
            if gt == 32000:
                total_labels.append(0)
                if pred == 32000:
                    total_prelabels.append(0)
                else:
                    total_prelabels.append(1)
            else:
                total_labels.append(1)
                if pred == gt:
                    total_prelabels.append(1)
                else:
                    total_prelabels.append(0)
            if pred == gt:
                Correct += 1
    
        
        total_prelabels = np.array(total_prelabels)
        total_labels = np.array(total_labels)
                
        classifi_report = classification_report(
            total_labels, total_prelabels, target_names=[0, 1], output_dict=True
        )
        positive_precision = classifi_report[1]["precision"]
        positive_recall = classifi_report[1]["recall"]
        positive_f1_score = classifi_report[1]["f1-score"]
        negative_precision = classifi_report[0]["precision"]
        negative_recall = classifi_report[0]["recall"]
        negative_f1 = classifi_report[0]["f1-score"]

        gold_tri_id_num, pred_tri_id_num, match_tri_id_num = 0, 0, 0
        
        # tri_id
        for gold_trigger, pred_trigger in zip(references, predictions):

            if gold_trigger == pred_trigger:
                match_tri_id_num += 1

            gold_tri_id_num += 1
            pred_tri_id_num += 1
        
        # tri_cls
        # gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num = 0, 0, 0
        # for gold_trigger, pred_trigger in zip(references, predictions):
        #     gold_set = set(gold_trigger)
        #     pred_set = set(pred_trigger)
        #     gold_tri_cls_num += len(gold_set)
        #     pred_tri_cls_num += len(pred_set)
        #     match_tri_cls_num += len(gold_set & pred_set)
            
        scores = {
            'tri_id': (gold_tri_id_num, pred_tri_id_num, match_tri_id_num) + self.compute_f1(pred_tri_id_num, gold_tri_id_num, match_tri_id_num),
            # 'tri_cls': (gold_tri_cls_num, pred_tri_cls_num, match_tri_cls_num) + self.compute_f1(pred_tri_cls_num, gold_tri_cls_num, match_tri_cls_num),
        }
                
        return {
            "accuracy": Correct / len(predictions),
            "Positive_Precision": positive_precision,
            "Positive_Recall": positive_recall,
            "Positive_F1": positive_f1_score,
            "Negative_Precision": negative_precision,
            "Negative_Recall": negative_recall,
            "Negative_F1": negative_f1,
            "gold_tri_id_num": scores['tri_id'][0],
            "pred_tri_id_num": scores['tri_id'][1],
            "match_tri_id_num": scores['tri_id'][2],
            "trriger_id_precision": scores['tri_id'][3],
            "trriger_id_recall": scores['tri_id'][4],
            "trriger_id_f1": scores['tri_id'][5],
            # "gold_tri_cls_num": scores['tri_cls'][0],
            # "pred_tri_cls_num": scores['tri_cls'][1],
            # "match_tri_cls_num": scores['tri_cls'][2],
            # "trriger_cls_precision": scores['tri_cls'][3],
            # "trriger_cls_recall": scores['tri_cls'][4],
            # "trriger_cls_f1": scores['tri_cls'][5],
            }
                
    def compute_f1(self, predicted, gold, matched):
        precision = self.safe_div(matched, predicted)
        recall = self.safe_div(matched, gold)
        f1 = self.safe_div(2 * precision * recall, precision + recall)
        return precision, recall, f1


    def safe_div(self, num, denom):
        if denom > 0:
            return num / denom
        else:
            return 0
        
    