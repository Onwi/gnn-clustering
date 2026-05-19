from functools import partial
from pathlib import Path
import pandas as pd
from sklearn.metrics import f1_score

from analyze_performance_results import build_coarsening_model_tables, build_df_coarsening_results, build_df_fixed_results, build_fixed_model_tables, plot_coarsening_results, plot_fixed_results


path_single_task_coarsening_sample_type_results = Path("/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_coarsening_sample_type_prediction")
path_single_task_coarsening_cohort_results = Path("/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_coarsening_cohort_classification")
path_multi_task_coarsening_results = Path("/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_multitask_coarsening_levels/")

path_single_task_fixed_sample_type_results = Path("/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_fixed_sample_type_prediction")
path_single_task_fixed_cohort_results = Path("/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_fixed_cohort_classification")
path_multi_task_fixed_results = Path("/home/thomas/scratch/tvfontanari/PoolingGenomicGNNsData/results_multitask_fixed_levels/")


class SingleTaskModelRunResults:
    def __init__(self, path_final_results: str) -> None:
        self.path_results = Path(path_final_results)
        self.path_predictions = self.path_results / 'predictions.csv'

    def compute_score(self, score_func):
        predictions = pd.read_csv(self.path_predictions, index_col=0)
        score = score_func(predictions['labels'], predictions['predictions'])
        return score


class MultiTaskModelRunResults:
    def __init__(self, path_final_results: str) -> None:
        self.path_results = Path(path_final_results)
        self.path_predictions_cohort = self.path_results / 'cohort_predictions.csv'
        self.path_predictions_type = self.path_results / 'type_predictions.csv'

    def compute_cohort_score(self, score_func):
        predictions = pd.read_csv(self.path_predictions_cohort, index_col=0)
        score = score_func(predictions['labels'], predictions['predictions'])
        return score

    def compute_sample_type_score(self, score_func):
        predictions = pd.read_csv(self.path_predictions_type, index_col=0)
        score = score_func(predictions['labels'], predictions['predictions'])
        return score


class SingleTaskCoarseningModelResults:
    def __init__(self, path_runs: str, n_levels: int, wpool: bool, convs: bool, n_holdouts: int = 5) -> None:
        self.n_levels = n_levels
        self.wpool = wpool
        self.convs = convs
        self.path_runs = Path(path_runs)
        self.n_holdouts = n_holdouts
    
    def compute_score(self, score_func):
        records = []
        for run in range(self.n_holdouts):
            run_result = SingleTaskModelRunResults(
                path_final_results=self.path_runs / self._model_dir_name(run) / "final_model_results"
            )
            score = run_result.compute_score(score_func)
            records.append({'run': run, 'score': score})

        scores = pd.DataFrame.from_records(records)
        return scores

    def _model_dir_name(self, run: int):
        return Path(f"nlevels{self.n_levels}_rep{run}_wpool{self.wpool}_convs{self.convs}")


class SingleTaskFixedModelResults:
    def __init__(self, path_runs: str, first_level: int, wpool: bool, n_holdouts: int = 5) -> None:
        self.first_level = first_level
        self.wpool = wpool
        self.path_runs = Path(path_runs)
        self.n_holdouts = n_holdouts
    
    def compute_score(self, score_func):
        records = []
        for run in range(self.n_holdouts):
            run_result = SingleTaskModelRunResults(
                path_final_results=self.path_runs / self._model_dir_name(run) / "final_model_results"
            )
            score = run_result.compute_score(score_func)
            records.append({'run': run, 'score': score})

        scores = pd.DataFrame.from_records(records)
        return scores

    def _model_dir_name(self, run: int):
        return Path(f"firstlevel{self.first_level}_rep{run}_wpool{self.wpool}_nlevels7")
    

class MultiTaskCoarseningModelResults:
    def __init__(self, path_runs: str, n_levels: int, wpool: bool, convs: bool, n_holdouts: int = 5) -> None:
        self.n_levels = n_levels
        self.wpool = wpool
        self.convs = convs
        self.path_runs = Path(path_runs)
        self.n_holdouts = n_holdouts
    
    def compute_score(self, score_func):
        records = []
        for run in range(self.n_holdouts):
            run_result = MultiTaskModelRunResults(
                path_final_results=self.path_runs / self._model_dir_name(run) / "final_model_results"
            )
            score_cohort = run_result.compute_cohort_score(score_func)
            score_sample_type = run_result.compute_sample_type_score(score_func)
            records.append({'run': run, 'score_cohort': score_cohort, 'score_sample_type': score_sample_type})

        scores = pd.DataFrame.from_records(records)
        return scores

    def _model_dir_name(self, run: int):
        return Path(f"tcga_cohort_and_tumor_nlevels{self.n_levels}_rep{run}_wpool{self.wpool}_convs{self.convs}")


class MultiTaskFixedModelResults:
    def __init__(self, path_runs: str, first_level: int, wpool: bool, n_holdouts: int = 5) -> None:
        self.first_level = first_level
        self.wpool = wpool
        self.path_runs = Path(path_runs)
        self.n_holdouts = n_holdouts
    
    def compute_score(self, score_func):
        records = []
        for run in range(self.n_holdouts):
            run_result = MultiTaskModelRunResults(
                path_final_results=self.path_runs / self._model_dir_name(run) / "final_model_results"
            )
            score_cohort = run_result.compute_cohort_score(score_func)
            score_sample_type = run_result.compute_sample_type_score(score_func)
            records.append({'run': run, 'score_cohort': score_cohort, 'score_sample_type': score_sample_type})

        scores = pd.DataFrame.from_records(records)
        return scores

    def _model_dir_name(self, run: int):
        return Path(f"tcga_cohort_and_tumor_firstlevel{self.first_level}_rep{run}_wpool{self.wpool}_nlevels7")


def build_score_table(single_task_cohort_results, single_task_sample_type_results, multitask_results):
    """Construct 2 4x4 tables comparing mean and std of multitask and singletask models"""
    scores_sample_type = single_task_sample_type_results.compute_score(partial(f1_score, average="macro"))
    scores_sample_type.rename(columns={'score': 'score_sample_type'}, inplace=True)
    print(scores_sample_type)

    scores_cohort = single_task_cohort_results.compute_score(partial(f1_score, average="macro"))
    scores_cohort.rename(columns={'score': 'score_cohort'}, inplace=True)
    print(scores_cohort)

    scores = multitask_results.compute_score(partial(f1_score, average="macro"))
    print(scores)

    data_mean = [
        [scores_sample_type['score_sample_type'].mean(), scores_cohort['score_cohort'].mean()],
        [scores['score_sample_type'].mean(), scores['score_cohort'].mean()]
    ]
    data_mean = pd.DataFrame(data=data_mean, columns=['Tumor Prediction', 'Cohort Prediction'], index=['Single-task', 'Multi-task'])

    data_std = [
        [scores_sample_type['score_sample_type'].std(), scores_cohort['score_cohort'].std()],
        [scores['score_sample_type'].std(), scores['score_cohort'].std()]
    ]
    data_std = pd.DataFrame(data=data_std, columns=['Tumor Prediction', 'Cohort Prediction'], index=['Single-task', 'Multi-task'])

    return data_mean, data_std


def build_nn_table():
    """Construct 2 4x4 tables comparing mean and std of multitask and singletask neural networks"""
    stc_res_sample_type = SingleTaskCoarseningModelResults(
        path_runs=path_single_task_coarsening_sample_type_results,
        n_levels=0,
        wpool=False,
        convs=False
    )

    stc_res_cohort = SingleTaskCoarseningModelResults(
        path_runs=path_single_task_coarsening_cohort_results,
        n_levels=0,
        wpool=False,
        convs=False
    )

    mtc_res = MultiTaskCoarseningModelResults(
        path_runs=path_multi_task_coarsening_results,
        n_levels=0,
        wpool=False,
        convs=False
    )

    build_score_table(single_task_cohort_results=stc_res_cohort, single_task_sample_type_results=stc_res_sample_type, multitask_results=mtc_res)


def test_build_fixed_results_table():
    """Construct 2 4x4 tables comparing mean and std of multitask and singletask neural networks"""
    stc_res_sample_type = SingleTaskFixedModelResults(
        path_runs=path_single_task_fixed_sample_type_results,
        first_level=4,
        wpool=True
    )

    stc_res_cohort = SingleTaskFixedModelResults(
        path_runs=path_single_task_fixed_cohort_results,
        first_level=4,
        wpool=True
    )

    mtc_res = MultiTaskFixedModelResults(
        path_runs=path_multi_task_fixed_results,
        first_level=4,
        wpool=True
    )

    df_mean, df_std = build_score_table(
        single_task_cohort_results=stc_res_cohort, 
        single_task_sample_type_results=stc_res_sample_type, 
        multitask_results=mtc_res
    )
    print(df_mean)
    print(df_std)


def mt_vs_st_coarsening_plots():
    """Difference plots between the multitask and singletask performances"""
    levels = [0, 1, 2, 3, 4, 5, 6]
    wpool_list = [True, False]
    use_convs_list = [True, False]
    reps = [0, 1, 2, 3, 4]
    df_records_st_sample_type, df_records_st_sample_type_nn = build_df_coarsening_results(
        path_results=Path(path_single_task_coarsening_sample_type_results),
        levels=levels,
        wpool_list=wpool_list,
        use_convs_list=use_convs_list,
        reps=reps,
        compute_metric="test_f1_macro",
        multitask_model=False
    )

    df_records_mt, df_records_mt_nn = build_df_coarsening_results(
        path_results=Path(path_multi_task_coarsening_results),
        levels=levels,
        wpool_list=wpool_list,
        use_convs_list=use_convs_list,
        reps=reps,
        compute_metric="test_f1_macro",
        multitask_model=True
    )

    ####
    # Plot results for sample_type
    df_a = df_records_mt.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro_type']]
    df_b = df_records_st_sample_type.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro']].rename(columns={'test_f1_macro': 'test_f1_macro_type'})
    df_diffs_coarsening = (df_a - df_b).reset_index()

    df_a_nn = df_records_mt_nn.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro_type']]
    df_b_nn = df_records_st_sample_type_nn.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro']].rename(columns={'test_f1_macro': 'test_f1_macro_type'})
    df_diffs_nn = (df_a_nn - df_b_nn).reset_index()

    output_dir = Path('tmp_output_mt_vs_st')
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_coarsening_results(
        df_records=df_diffs_coarsening,
        df_records_nn=df_diffs_nn,
        n_levels_list=levels,
        metric='test_f1_macro_type',
        output_dir=output_dir,
        ylabel='$MT_{F1} - ST_{F1}$'
    )

    df_summary = build_coarsening_model_tables(
        df_records=df_diffs_coarsening,
        df_records_nn=df_diffs_nn,
        output_dir=output_dir,
        suffix="_diffs_sampletype"
    )

    ####
    # Plot results for cohort classification
    df_records_st_cohort, df_records_st_cohort_nn = build_df_coarsening_results(
        path_results=Path(path_single_task_coarsening_cohort_results),
        levels=levels,
        wpool_list=wpool_list,
        use_convs_list=use_convs_list,
        reps=reps,
        compute_metric="test_f1_macro",
        multitask_model=False
    )

    df_a = df_records_mt.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro']]
    df_b = df_records_st_cohort.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro']]
    df_diffs_coarsening = (df_a - df_b).reset_index()

    df_a_nn = df_records_mt_nn.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro']]
    df_b_nn = df_records_st_cohort_nn.set_index(['n_levels', 'wpool', 'convs', 'rep'])[['test_f1_macro']]
    df_diffs_nn = (df_a_nn - df_b_nn).reset_index()
    
    output_dir = Path('tmp_output_mt_vs_st')
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_coarsening_results(
        df_records=df_diffs_coarsening,
        df_records_nn=df_diffs_nn,
        n_levels_list=levels,
        metric='test_f1_macro',
        output_dir=output_dir,
        ylabel='$MT_{F1} - ST_{F1}$'
    )

    df_summary = build_coarsening_model_tables(
        df_records=df_diffs_coarsening,
        df_records_nn=df_diffs_nn,
        output_dir=output_dir,
        suffix="_diff_cohorts"
    )


def mt_vs_st_fixed_plots():
    """Difference plots between the multitask and singletask performances"""
    first_levels = [0, 1, 2, 3, 4, 5, 6, 7]
    wpool_list = [True, False]
    reps = [0, 1, 2, 3, 4]
    n_levels = 7
    df_records_st_sample_type = build_df_fixed_results(
        path_results=Path(path_single_task_fixed_sample_type_results),
        first_levels=first_levels,
        wpool_list=wpool_list,
        reps=reps,
        n_levels=n_levels, 
        compute_metric="test_f1_macro",
        multitask_model=False
    )

    df_records_mt = build_df_fixed_results(
        path_results=Path(path_multi_task_fixed_results),
        first_levels=first_levels,
        wpool_list=wpool_list,
        reps=reps,
        n_levels=n_levels, 
        compute_metric="test_f1_macro",
        multitask_model=True
    )

    columns = ['first_level', 'wpool', 'rep']

    ###
    # Plot results for sample_type
    df_a = df_records_mt.set_index(columns)[['test_f1_macro_type']]
    df_b = df_records_st_sample_type.set_index(columns)[['test_f1_macro']].rename(columns={'test_f1_macro': 'test_f1_macro_type'})
    df_diffs = (df_a - df_b).reset_index()

    output_dir = Path('tmp_output_mt_vs_st')
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_fixed_results(
        df_records=df_diffs,
        n_levels=n_levels,
        metric='test_f1_macro_type',
        output_dir=output_dir,
        ylabel='$MT_{F1} - ST_{F1}$',
        ylim=[-0.025, 0.025]
    )

    ####
    # Plot results for cohort classification
    df_records_st_cohort = build_df_fixed_results(
        path_results=Path(path_single_task_fixed_cohort_results),
        first_levels=first_levels,
        wpool_list=wpool_list,
        reps=reps,
        n_levels=n_levels, 
        compute_metric="test_f1_macro",
        multitask_model=False
    )

    df_a = df_records_mt.set_index(columns)[['test_f1_macro']]
    df_b = df_records_st_cohort.set_index(columns)[['test_f1_macro']]
    df_diffs = (df_a - df_b).reset_index()
    
    output_dir = Path('tmp_output_mt_vs_st')
    output_dir.mkdir(exist_ok=True, parents=True)
    plot_fixed_results(
        df_records=df_diffs,
        n_levels=n_levels,
        metric='test_f1_macro',
        output_dir=output_dir,
        ylabel='$MT_{F1} - ST_{F1}$',
        ylim=[-0.05, 0.05]
    )

    df_summary = build_fixed_model_tables(
        df_records=df_diffs,
        output_dir=output_dir
    )


def main():
    mt_vs_st_coarsening_plots()
    mt_vs_st_fixed_plots()


if __name__ == "__main__":
    main()
