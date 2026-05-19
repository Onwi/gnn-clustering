import argparse
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, f1_score


metric_label_map = {
    "test_accuracy": "Test Accuracy",
    "test_accuracy_type": "Test Accuracy",
    "test_balanced_accuracy": "Test Balanced Accuracy",
    "test_balanced_accuracy_type": "Test Balanced Accuracy",
    "test_f1_macro": "Test F1 Macro Average",
    "test_f1_macro_type": "Test F1 Macro Average",
}
predictions_fname = 'predictions.csv'
cohort_predictions_fname = 'cohort_predictions.csv'
type_predictions_fname = 'type_predictions.csv'

metrics_of_interst = ["test_f1_macro", "test_accuracy", "test_balanced_accuracy"]
metric_choices = metrics_of_interst + ["all"]


def get_metric_label(metric: str):
    return metric_label_map[metric]


def plot_coarsening_results(
    df_records,
    df_records_nn,
    n_levels_list,
    show: bool = False,
    metric: str = "test_accuracy",
    output_dir: Path = None,
    ylim = None,
    ylabel: str = None
):
    df_records_nn_repeated = pd.DataFrame()
    fig, ax = plt.subplots()
    for nlvls in n_levels_list[1:]:
        df_records_nn_nlvls = df_records_nn.copy()
        df_records_nn_nlvls['n_levels'] = nlvls
        df_records_nn_repeated = pd.concat((df_records_nn_repeated, df_records_nn_nlvls), axis=0)
    df_records_nn_repeated.reset_index(inplace=True, drop=True)
    df_records_nn_repeated.to_csv('tmp_df_records_nn.csv')

    sns.lineplot(
        data=df_records_nn_repeated, 
        x="n_levels", y=metric, ax=ax, linestyle='--', label='Neural Network',
        color='tab:purple'
    )
    sns.lineplot(
        data=df_records[(df_records['wpool'] == True) & (df_records['convs'] == True)], 
        x="n_levels", y=metric, ax=ax, linestyle='-', label='Weighted Pooling + Convs',
        color='tab:blue'
    )
    sns.lineplot(
        data=df_records[(df_records['wpool'] == True) & (df_records['convs'] == False)], 
        x="n_levels", y=metric, ax=ax, linestyle='--', label='Only Weighted Pooling',
        color='tab:orange'
    )
    sns.lineplot(
        data=df_records[(df_records['wpool'] == False) & (df_records['convs'] == True)], 
        x="n_levels", y=metric, ax=ax, linestyle='-', label='Sum Pooling + Convs',
        color='tab:green'
    )
    sns.lineplot(
        data=df_records[(df_records['wpool'] == False) & (df_records['convs'] == False)], 
        x="n_levels", y=metric, ax=ax, linestyle='--', label='Only Sum Pooling',
        color='tab:red'
    )
    if 0 in n_levels_list:
        ax.set_xticks(n_levels_list[1:])
    else:
        ax.set_xticks(n_levels_list)

    ax.set_xlabel("Coarsening Levels")
    if ylabel is None:
        ax.set_ylabel(get_metric_label(metric))
    else:
        ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid()

    fig.tight_layout()

    if show:
        plt.show()

    if output_dir:
        path_output = output_dir / f"coarsening_{metric}.jpg"
        fig.savefig(str(path_output))

        path_output = output_dir / f"coarsening_{metric}.pdf"
        fig.savefig(str(path_output))

        plt.close(fig)


def build_df_coarsening_results(
    path_results, levels, wpool_list, use_convs_list, reps, compute_metric: str = None, multitask_model: bool = False
):
    records = []
    for rep in reps:
        for wpool in wpool_list:
            for n_levels in levels:
                for convs in use_convs_list:
                    if n_levels == 0:
                        if wpool == True or convs == True:
                            continue
                    if multitask_model:
                        dir_name = f"tcga_cohort_and_tumor_nlevels{n_levels}_rep{rep}_wpool{wpool}_convs{convs}"
                    else:
                        dir_name = f"nlevels{n_levels}_rep{rep}_wpool{wpool}_convs{convs}"

                    test_results = path_results / dir_name / "final_model_results"
                    metrics_path = test_results / "metrics.csv"

                    metrics = pd.read_csv(metrics_path, index_col=0).iloc[-1, :]
                    record = metrics.to_dict()
                    record["n_levels"] = n_levels
                    record["wpool"] = wpool
                    record["convs"] = convs
                    record["rep"] = rep
                    if compute_metric is not None:
                        if compute_metric not in record:
                            print(f"Metric {compute_metric} not in pre-computed values. It will be computed now.")
                            fname = cohort_predictions_fname if multitask_model else predictions_fname
                            mv = compute_metric_value(
                                compute_metric, test_results / f"{fname}"
                            )
                            record[f"{compute_metric}"] = mv
                        if multitask_model:
                            if f"{compute_metric}_type" not in record:
                                print(f"Metric {compute_metric} for type classification not in pre-computed values. It will be computed now.")
                                mv = compute_metric_value(
                                    compute_metric, test_results / f"{type_predictions_fname}"
                                )
                                record[f"{compute_metric}_type"] = mv

                    records.append(record)

    df_records = pd.DataFrame.from_records(records)
    df_records["n_levels"] = df_records["n_levels"].astype(int)
    df_records_nn = df_records[df_records["n_levels"] == 0].copy()
    df_records = df_records[df_records["n_levels"] != 0]
    return df_records, df_records_nn


def compute_metric_value(metric: str, path_predictions: Path):
    df = pd.read_csv(path_predictions)
    y_pred = df["predictions"]
    y_true = df["labels"]
    if "f1_macro" in metric:
        mv = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    elif "balanced_accuracy" in metric:
        mv = balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
    return mv


def build_df_fixed_results(
    path_results, first_levels, wpool_list, reps, n_levels, compute_metric: str = None, 
    multitask_model: bool = False
):
    records = []
    for rep in reps:
        for wpool in wpool_list:
            for first_level in first_levels:
                if multitask_model:
                    dir_name = f"tcga_cohort_and_tumor_firstlevel{first_level}_rep{rep}_wpool{wpool}_nlevels{n_levels}"
                else:
                    dir_name = (
                        f"firstlevel{first_level}_rep{rep}_wpool{wpool}_nlevels{n_levels}"
                    )
                test_results = path_results / dir_name / "final_model_results"
                metrics_path = test_results / "metrics.csv"

                metrics = pd.read_csv(metrics_path, index_col=0).iloc[-1, :]
                record = metrics.to_dict()
                record["first_level"] = first_level + 1
                record["wpool"] = wpool
                record["rep"] = rep
                if compute_metric is not None:
                    if compute_metric not in record:
                        print(f"Metric {compute_metric} not in pre-computed values. It will be computed now.")
                        fname = cohort_predictions_fname if multitask_model else predictions_fname
                        mv = compute_metric_value(
                            compute_metric, test_results / f"{fname}"
                        )
                        record[f"{compute_metric}"] = mv
                    if multitask_model:
                        if f"{compute_metric}_type" not in record:
                            print(f"Metric {compute_metric} for type classification not in pre-computed values. It will be computed now.")
                            mv = compute_metric_value(
                                compute_metric, test_results / f"{type_predictions_fname}"
                            )
                            record[f"{compute_metric}_type"] = mv

                records.append(record)

    df_records = pd.DataFrame.from_records(records)
    df_records["first_level"] = df_records["first_level"].astype(int)
    return df_records


def plot_fixed_results(
    df_records,
    n_levels,
    show: bool = False,
    metric: str = "test_accuracy",
    output_dir: Path = None,
    ylim=None,
    ylabel=None
):
    df_hline = (
        df_records[df_records["first_level"] == n_levels + 1]
        .copy()
        .drop(columns="first_level")
    )
    dfs = []

    for fl in range(df_records["first_level"].min(), df_records["first_level"].max()):
        df = df_hline.copy()
        df["first_level"] = fl
        df["hline"] = True
        dfs.append(df)

    df_hline = pd.concat(dfs, axis=0).reset_index(drop=True)
    df_records_w_hline = pd.concat([df_records, df_hline], axis=0, ignore_index=True)
    
    def make_hue(row):
        if pd.isna(row.hline):
            if row.wpool == True:
                return "Weighted Pooling + Convs"
            else:
                return "Sum Pooling + Convs"
        else:
            if row.wpool == True:
                return "Weighted Pooling - No Convs"
            else:
                return "Sum Pooling - No Convs"

    hue = df_records_w_hline[["wpool", "hline"]].apply(make_hue, axis=1)

    df_records_w_hline["hue"] = hue
    df_record_convs = df_records_w_hline[pd.isna(df_records_w_hline["hline"])]
    df_record_no_convs = df_records_w_hline[pd.notna(df_records_w_hline["hline"])]
    print(df_record_convs)
    fig, ax = plt.subplots()
    sns.lineplot(
        data=df_record_convs[
            (df_record_convs['wpool'] == True) & (df_record_convs["first_level"] != 8)
        ], 
        y=metric, x='first_level', ax=ax, linestyle='-', label='Weighted Pooling + Convs',
        color='tab:blue'
    )
    sns.lineplot(
        data=df_record_no_convs[df_record_no_convs['wpool'] == True], 
        y=metric, x='first_level', ax=ax, linestyle='--', label='Only Weighted Pooling',
        color='tab:orange'
    )
    sns.lineplot(
        data=df_record_convs[
            (df_record_convs['wpool'] == False) & (df_record_convs["first_level"] != 8)
        ],
        y=metric, x='first_level', ax=ax, linestyle='-', label='Sum Pooling + Convs',
        color='tab:green'
    )
    sns.lineplot(
        data=df_record_no_convs[df_record_no_convs['wpool'] == False], 
        y=metric, x='first_level', ax=ax, linestyle='--', label='Only Sum Pooling',
        color='tab:red'
    )
    ax.legend()
    ax.set_xlabel("First Level with GNN")
    if ylabel is None:
        ax.set_ylabel(get_metric_label(metric))
    else:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid()

    fig.tight_layout()
    
    if show:
        plt.show()

    if output_dir:
        path_output = output_dir / f"fixed_{metric}.jpg"
        fig.savefig(str(path_output))

        path_output = output_dir / f"fixed_{metric}.pdf"
        fig.savefig(str(path_output))

        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=["coarsening", "fixed"])
    parser.add_argument(
        "--path-results",
        type=str,
        default="/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_coarsening_levels_brca",
    )
    parser.add_argument("--levels", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6])
    parser.add_argument(
        "--first-levels", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7]
    )
    parser.add_argument(
        "--weighted-pooling", nargs="+", type=bool, default=[True, False]
    )
    parser.add_argument("--use-convs", nargs="+", type=bool, default=[True, False])
    parser.add_argument("--reps", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--metric", type=str, default="test_accuracy", choices=metric_choices
    )
    parser.add_argument("--output-dir", type=str, default="analysis_outputs")
    parser.add_argument("--show-figures", action='store_true')
    parser.add_argument("--n-levels", type=int, default=7)
    parser.add_argument('--ylim', type=float, nargs=2, help="y-axis limits")
    parser.add_argument('--ylim-type', type=float, nargs=2, help="y-axis limits for the type task, if multitasking")
    args = parser.parse_args()
    return args


def make_coarsening_model_name(row):
    if row.convs:
        if row.wpool:
            return 'Weighted Pooling + Convs'
        else:
            return 'Sum Pooling + Convs'
    else:
        if row.wpool:
            return 'Only Weighted Pooling'
        else:
            return 'Only Sum Pooling'


def make_fixed_model_name(row, no_convs_level: int = 8):
    if row.first_level != no_convs_level:
        if row.wpool:
            return 'Weighted Pooling + Convs'
        else:
            return 'Sum Pooling + Convs'
    else:
        if row.wpool:
            return 'Only Weighted Pooling'
        else:
            return 'Only Sum Pooling'


def build_coarsening_model_tables(
    df_records: pd.DataFrame,
    df_records_nn: pd.DataFrame,
    output_dir: Path = None, 
    suffix: str = ""
):
    def func_rename_column(col: str):
        if col == 'n_levels':
            return 'Coarsening Levels'
        if '_' in col:
            return col.replace('_', ' ').title()
        return col

    # Compute metrics summary for the GNN coarsening models
    df_records = df_records.drop(columns=['epoch', 'rep'], errors='ignore')
    df_records['Model'] = df_records[['wpool', 'convs']].apply(make_coarsening_model_name, axis=1)
    df_records = df_records.drop(columns=['wpool', 'convs'], errors='ignore')
    df_records = df_records.rename(columns=func_rename_column)
    df_summary = df_records.groupby(
        by=['Coarsening Levels', 'Model']
    ).agg(['min', 'max', 'median', 'mean', 'std']).reset_index()

    # Compute metrics summary for the neural network
    df_records_nn = df_records_nn.drop(columns=['epoch', 'rep', 'wpool', 'convs', 'n_levels'], errors='ignore')
    df_records_nn = df_records_nn.rename(columns=func_rename_column)
    df_records_nn['Model'] = 'Neural Network'
    df_summary_nn = df_records_nn.groupby(by=['Model']).agg(['min', 'max', 'median', 'mean', 'std']).reset_index()
    df_summary = pd.concat((df_summary, df_summary_nn), axis=0)
    
    if output_dir is not None:
        df_summary_csv = df_summary.set_index(['Coarsening Levels', 'Model'])
        for g, df_g in df_summary_csv.groupby(level=0, axis=1):
            df_g.reset_index().to_csv(output_dir / f"{g}_results_summary{suffix}.csv", index=False)
    
    return df_summary
    

def run_coarsening_analysis(
    output_dir, path_results, levels, wpool_list, use_convs_list, reps, metric, show, ylim, ylim_type=None
):
    exp_output_dir = output_dir / f"analysis_{Path(path_results).name}"
    exp_output_dir.mkdir(exist_ok=True, parents=True)
    multitask = ('multitask' in path_results)

    df_records, df_records_nn = build_df_coarsening_results(
        path_results=Path(path_results),
        levels=levels,
        wpool_list=wpool_list,
        use_convs_list=use_convs_list,
        reps=reps,
        compute_metric=metric,
        multitask_model=multitask
    )
    df_summary = build_coarsening_model_tables(
        df_records=df_records,
        df_records_nn=df_records_nn,
        output_dir=exp_output_dir
    )

    if multitask:
        plot_coarsening_results(
            df_records=df_records,
            df_records_nn=df_records_nn,
            n_levels_list=levels,
            metric=f"{metric}_type",
            output_dir=exp_output_dir,
            show=show,
            ylim=ylim_type
        )
        plot_coarsening_results(
            df_records=df_records,
            df_records_nn=df_records_nn,
            n_levels_list=levels,
            metric=f"{metric}",
            output_dir=exp_output_dir,
            show=show,
            ylim=ylim
        )
    else:
        plot_coarsening_results(
            df_records=df_records,
            df_records_nn=df_records_nn,
            n_levels_list=levels,
            metric=metric,
            output_dir=exp_output_dir,
            show=show,
            ylim=ylim
        )


def build_fixed_model_tables(
    df_records: pd.DataFrame,
    output_dir: Path = None
):
    def func_rename_column(col: str):
        if col == 'first_level':
            return 'First Level with GNN'
        if '_' in col:
            return col.replace('_', ' ').title()
        return col

    print(df_records)
    df_records = df_records.drop(columns=['epoch', 'rep'])
    df_records['Model'] = df_records[['wpool', 'first_level']].apply(make_fixed_model_name, axis=1)
    df_records = df_records.drop(columns=['wpool'])
    df_records = df_records.rename(columns=func_rename_column)

    df_summary = df_records.groupby(
        by=['First Level with GNN', 'Model']
    ).agg(['min', 'max', 'median', 'mean', 'std']).reset_index()
    df_summary['First Level with GNN'] = df_summary['First Level with GNN'].replace({8: pd.NA})
    
    if output_dir is not None:
        df_summary_csv = df_summary.set_index(['First Level with GNN', 'Model'])
        for g, df_g in df_summary_csv.groupby(level=0, axis=1):
            df_g.reset_index().to_csv(output_dir / f"{g}_results_summary.csv", index=False)
    return df_summary


def run_fixed_analysis(
    output_dir, path_results, first_levels, n_levels, wpool_list, reps, metric, show, ylim, ylim_type=None
):
    exp_output_dir = output_dir / f"analysis_{Path(path_results).name}"
    exp_output_dir.mkdir(exist_ok=True, parents=True)
    multitask = ('multitask' in path_results)

    df_records = build_df_fixed_results(
        path_results=Path(path_results),
        first_levels=first_levels,
        wpool_list=wpool_list,
        reps=reps,
        n_levels=n_levels,
        compute_metric=metric,
        multitask_model=multitask
    )
    df_summary = build_fixed_model_tables(
        df_records=df_records,
        output_dir=exp_output_dir
    )
    if multitask:
        plot_fixed_results(
            df_records=df_records,
            n_levels=n_levels,
            metric=metric,
            output_dir=exp_output_dir,
            show=show,
            ylim=ylim
        )
        plot_fixed_results(
            df_records=df_records,
            n_levels=n_levels,
            metric=f"{metric}_type",
            output_dir=exp_output_dir,
            show=show,
            ylim=ylim_type
        )
    else:
        plot_fixed_results(
            df_records=df_records,
            n_levels=n_levels,
            metric=metric,
            output_dir=exp_output_dir,
            show=show,
            ylim=ylim
        )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.experiment == "coarsening":
        if args.metric != "all":
            run_coarsening_analysis(
                output_dir=output_dir,
                path_results=args.path_results,
                levels=args.levels,
                wpool_list=args.weighted_pooling,
                use_convs_list=args.use_convs,
                reps=args.reps,
                metric=args.metric,
                show=args.show_figures,
                ylim=args.ylim,
                ylim_type=args.ylim_type
            )
        else:
            for metric in metrics_of_interst:
                run_coarsening_analysis(
                    output_dir=output_dir,
                    path_results=args.path_results,
                    levels=args.levels,
                    wpool_list=args.weighted_pooling,
                    use_convs_list=args.use_convs,
                    reps=args.reps,
                    metric=metric,
                    show=args.show_figures,
                    ylim=args.ylim,
                    ylim_type=args.ylim_type
                )

    if args.experiment == "fixed":
        if args.metric != "all":
            run_fixed_analysis(
                output_dir=output_dir,
                path_results=args.path_results,
                first_levels=args.first_levels,
                wpool_list=args.weighted_pooling,
                n_levels=args.n_levels,
                reps=args.reps,
                metric=args.metric,
                show=args.show_figures,
                ylim=args.ylim,
                ylim_type=args.ylim_type
            )
        else:
            for metric in metrics_of_interst:
                run_fixed_analysis(
                    output_dir=output_dir,
                    path_results=args.path_results,
                    first_levels=args.first_levels,
                    wpool_list=args.weighted_pooling,
                    n_levels=args.n_levels,
                    reps=args.reps,
                    metric=metric,
                    show=args.show_figures,
                    ylim=args.ylim,
                    ylim_type=args.ylim_type
                )


if __name__ == "__main__":
    main()
