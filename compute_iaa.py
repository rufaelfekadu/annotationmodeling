import experiments
import pandas as pd
import numpy as np
import json
from agreement import InterAnnotatorAgreement
from granularity import SeqRange

LABELS = ['SR','ISR','MUR','P','B', 'V', 'FG', 'HM', 'ME']
agg_item = {x:'max' for x in LABELS}
agg_ann = {x:'sum' for x in LABELS}
def merge_duplicates(df):
   # Group by item and annotator
   grouped = df.groupby(['item', 'annotator']).agg({
       'media_file': 'first',  # Keep first occurrence
       'start': 'min',
       'end': 'max',
       **agg_item,
   }).reset_index()
   
   grouped = grouped.groupby('item').agg({
         'media_file': 'first',  # Keep first occurrence
         'start': 'min',
         'end': 'max',
         **agg_ann,
    }).reset_index()
   
   return grouped

def binary_distance(x, y):
    return 1 if x != y else 0

# euclidian distance
def euclidian_distance(a1, a2, max_value=3):
    return abs(a1 - a2) / max_value

def iou(vrA, vrB):
    xA = max(vrA.start_vector[0], vrB.start_vector[0])
    xB = min(vrA.end_vector[0], vrB.end_vector[0])

    interrange = max(0, xB - xA + 1) 
    unionrange = (vrA.end_vector[0] - vrA.start_vector[0] + 1) + (vrB.end_vector[0] - vrB.start_vector[0] + 1) - interrange
    return (interrange / unionrange)

def inverse_iou(vrA, vrB):
    return 1 - iou(vrA, vrB)

def main(args):
    # compute iou for each class
    grannodf = pd.read_csv(args.input_path)
    results = {}
    grannodf = grannodf[~grannodf['annotator'].isin(['Gold','bau','mas','sad'])]
    # grannodf['item'] = grannodf['item'].astype('category').cat.codes
    # grannodf['annotator'] = grannodf['annotator'].astype('category').cat.codes
    print(grannodf['annotator'].unique())
    grannodf = merge_duplicates(grannodf)
    grannodf.to_csv(f'{args.save_path}/merged_annotations.csv', index=False)
    breakpoint()
    # print the count of annotattions with value greater than 1 for each class
    counts_dict = {}
    for label in LABELS:
        counts_dict[label] = {
            "agreement_1": int(grannodf[grannodf[label] >= 2][label].count()),
            "agreement_0": int(grannodf[grannodf[label] == 0][label].count()),
            "disagreement": int(grannodf[grannodf[label] == 1][label].count()),
            "total": int(grannodf[label].count())
        }
    with open(f'{args.save_path}/label_counts.json', 'w') as f:
        json.dump(counts_dict, f, indent=4)
    
    breakpoint()
    grannodf['timevr'] = grannodf[['start','end']].apply(lambda row: SeqRange(row.to_list()), axis=1)
    
    # compute IAA for each class
    for label in LABELS[:-1]:
        iaa = InterAnnotatorAgreement(grannodf, 
                                      item_colname=args.item_col, 
                                      uid_colname=args.annotator_col, 
                                      label_colname=label,
                                      distance_fn=binary_distance)
        iaa.setup()
        results[label] = {
            'alpha': iaa.get_krippendorff_alpha(),
            'ks': iaa.get_ks(),
            'sigma': iaa.get_sigma(use_kde=False),
        }
        # average  number of annotaions per annotator for a given class
        results[label]['num_annotations'] = grannodf[grannodf[label] == 1].groupby(args.annotator_col).size().mean()

    # compute IAA for tension
    iaa = InterAnnotatorAgreement(grannodf, 
                                  item_colname=args.item_col, 
                                  uid_colname=args.annotator_col, 
                                  label_colname='T',
                                  distance_fn=euclidian_distance)
    iaa.setup()
    results['T'] = {
        'alpha': iaa.get_krippendorff_alpha(),
        'ks': iaa.get_ks(),
        'sigma': iaa.get_sigma(use_kde=False),
        'num_annotations': grannodf[grannodf['T'] >= 1].groupby(args.annotator_col).size().mean()
    }

    # compute IAA for bounding boxes
    iaa = InterAnnotatorAgreement(grannodf, 
                                  item_colname=args.item_col, 
                                  uid_colname=args.annotator_col, 
                                  label_colname='timevr',
                                  distance_fn=inverse_iou)
    iaa.setup()
    results['bbox'] = {
        'alpha': iaa.get_krippendorff_alpha(),
        'ks': iaa.get_ks(),
        'sigma': iaa.get_sigma(use_kde=False),
        'num_annotations': grannodf.groupby(args.annotator_col).size().mean()
    }


    with open(f'{args.save_path}/iaa_results.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='../Stutter-Detection-Dataset/data/fluencybank_aws/interview/total_dataset_final.csv')
    parser.add_argument('--save_path', type=str, default='../Stutter-Detection-Dataset/data/fluencybank_aws/interview')
    parser.add_argument('--item_col', type=str, default='item')
    parser.add_argument('--annotator_col', type=str, default='annotator')
    args = parser.parse_args()
    main(args)
    

