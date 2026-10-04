import pandas as pd
import pickle
import os,sys
import argparse
sys.path.append("..")
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
from tqdm import tqdm
from CadSeqProc.cad_sequence import CADSequence
from CadSeqProc.utility.utils import (create_path_with_time,ensure_dir)
from CadSeqProc.utility.logger import CLGLogger
from tqdm import tqdm
import traceback
from rich import print
import json

csnLogger=CLGLogger().configure_logger().logger


def main():
    parser=argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input_path",help="Predicted CAD Sequence in pkl format",required=True)
    parser.add_argument("--output_dir",help="Output dir",required=True)
    parser.add_argument("--verbose",action='store_true')

    args=parser.parse_args()
    output_dir=create_path_with_time(args.output_dir)
    

    if args.verbose:
        csnLogger.info("Evaluation for Design History")
        csnLogger.info(f"Output Path {output_path}")

    with open(args.input_path,"rb") as f:
        data=pickle.load(f)
    
    for level in range(1,5):
        csnLogger.info(f"Level {level}")
        output_path=os.path.join(output_dir,'level_'+str(level))
        ensure_dir(output_path)
        generate_analysis_report(data=data,output_path=output_path,
                                logger=csnLogger,verbose=args.verbose, level='level_'+str(level))


def generate_analysis_report(data,output_path,logger,verbose,level):
    report_df = pd.DataFrame() # Dataframe for analysis
    # cm=np.zeros((4,4)) # Confusion Matrix

    uids=list(data.keys())

    for uid in tqdm(uids):
        best_report_df=process_uid_(uid,data,level=level)
        if best_report_df is not None:
            report_df=pd.concat([report_df,best_report_df])
    csv_path=os.path.join(output_path,f"report_df_{level}.csv")

    try:
        report_df.to_csv(csv_path, index=None)
        # logger.success(f"Report is saved at {csv_path}")
    except Exception as e:
        logger.error(f"Error saving csv file at {csv_path}")
        if verbose:
           print(traceback.print_exc())

    if verbose:
        logger.info("Calculating Metrics...")

    eval_dict = {}

    line_metrics = report_df[(report_df['line_total_gt'] > 0)][['line_recall', 'line_precision', 'line_f1']].mean() * 100
    eval_dict['line'] = {
        'recall': line_metrics['line_recall'],
        'precision': line_metrics['line_precision'],
        'f1': line_metrics['line_f1']
    }

    # Mean Recall, Precision, F1 for Arc
    arc_metrics = report_df[(report_df['arc_total_gt'] > 0)][['arc_recall', 'arc_precision', 'arc_f1']].mean() * 100
    eval_dict['arc'] = {
        'recall': arc_metrics['arc_recall'],
        'precision': arc_metrics['arc_precision'],
        'f1': arc_metrics['arc_f1']
    }

    # Mean Recall, Precision, F1 for Circle
    circle_metrics = report_df[(report_df['circle_total_gt'] > 0)][['circle_recall', 'circle_precision', 'circle_f1']].mean() * 100
    eval_dict['circle'] = {
        'recall': circle_metrics['circle_recall'],
        'precision': circle_metrics['circle_precision'],
        'f1': circle_metrics['circle_f1']
    }

    # Mean Recall, Precision, F1 for Extrusion
    ext_recall = report_df['num_ext'] / report_df['num_ext_gt']
    ext_precision = report_df['num_ext'] / report_df['num_ext_pred']
    ext_f1 = 2 * ext_recall * ext_precision / (ext_recall + ext_precision)
    extrusion_metrics = {
        'recall': ext_recall.mean() * 100,
        'precision': ext_precision.mean() * 100,
        'f1': ext_f1.mean() * 100
    }
    eval_dict.update({'extrusion': extrusion_metrics})

    
    # Update Chamfer Distance
    eval_dict['cd']={}
    eval_dict['cd']['median']=report_df['cd'][report_df['cd']>0].median()
    eval_dict['cd']['mean']=report_df['cd'][report_df['cd']>0].mean()
    eval_dict['invalidity_ratio_percentage']=report_df['cd'][report_df['cd']<0].count()*100/len(report_df)

    if verbose:
        json_formatted_str = json.dumps(eval_dict, indent=4)
        print(json_formatted_str)

    mean_report_path=os.path.join(output_path,f"mean_report_{level}.json")

    with open(mean_report_path,"w") as f:
        json.dump(eval_dict,f, indent=4)



def process_vec(pred_vec,gt_vec,bit,uid):
    try:
        pred_cad=CADSequence.from_vec(pred_vec,8,denumericalize=False)
        gt_cad=CADSequence.from_vec(gt_vec,8,denumericalize=False)

        report_df,cm=gt_cad.generate_report(pred_cad,uid)
        
        return report_df,cm
    except Exception as e:
        #print(e)
        return None,None

def process_uid_(uid,data,level):
    try:
        gt_vec = data[uid][level]['gt_cad_vec']
        all_cd = data[uid][level]['cd']
        best_index = 0
        pred_vec = data[uid][level]['pred_cad_vec'][best_index]
        df, _ = process_vec(pred_vec, gt_vec, 8, uid)
        df['cd'] = all_cd[best_index]

        return df

    except Exception as e:
        return None

if __name__=="__main__":
    main()