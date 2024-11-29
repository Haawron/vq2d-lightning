import json
from pathlib import Path

import torch

import lightning as L

from lightning.pytorch.callbacks import BasePredictionWriter

from ltvu.utils.compute_results import get_final_preds_vq2d, fix_predictions_order
from ltvu.utils.metrics import get_metrics_vq2d, format_metrics_vq2d, get_metrics_lasot, format_metrics_lasot


class PerSegmentWriter(BasePredictionWriter):
    """
    Callback to handle per-segment prediction writing during prediction epochs.
    This callback merges predictions from distributed ranks, processes them, and writes
    the results to intermediate and final JSON files.

    Attributes
    ----------
    output_dir : str or pathlib.Path
        Path to the directory where prediction results will be stored.
    official_anns_dir : str or pathlib.Path
        Path to the directory containing official annotations.
    test_submit : bool
        Whether the predictions are for test submission.
    movement : str
        Additional identifier for movement-related settings.
    split : str
        Dataset split being processed ('val' or 'test_unannotated').
    p_outdir : pathlib.Path
        Path to the main output directory.
    p_tmp_outdir : pathlib.Path
        Path to the temporary directory for storing intermediate rank outputs.
    p_pred : pathlib.Path
        Path to the final predictions JSON file.
    p_int_pred : pathlib.Path
        Path to the intermediate predictions file.
    p_metrics : pathlib.Path
        Path to the metrics JSON file.
    p_metrics_log : pathlib.Path
        Path to the metrics log file.
    rank_seg_preds : list
        List of predictions collected from the current rank.
    """
    def __init__(self,
        output_dir,
        official_anns_dir,
        test_submit = False,
        movement = "",
    ):
        """
        Initialize the PerSegmentWriter callback.

        Parameters
        ----------
        output_dir : str or pathlib.Path
            Path to the output directory.
        official_anns_dir : str or pathlib.Path
            Path to the official annotations directory.
        test_submit : bool, optional
            Whether the predictions are for test submission, by default False.
        movement : str, optional
            Additional identifier for movement-related settings, by default "".
        """
        super().__init__(write_interval="batch")
        self.p_outdir = Path(output_dir)
        self.p_int_pred = self.p_outdir / 'intermediate_predictions.pt'
        self.p_pred = self.p_outdir / 'predictions.json'
        self.p_metrics = self.p_outdir / 'metrics.json'
        self.p_metrics_log = self.p_outdir / 'metrics.log'
        self.rank_seg_preds = []
        self.test_submit = test_submit
        self.official_anns_dir = Path(official_anns_dir)
        self.movement = movement

        if self.test_submit:
            self.split = 'test_unannotated'
            self.p_pred = self.p_pred.with_name('test_predictions.json')
            self.p_int_pred = self.p_int_pred.with_name('test_intermediate_predictions.pt')
        else:
            self.split = 'val'

        self.p_tmp_outdir = self.p_outdir / 'tmp' / self.split

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        """
        Setup temporary directories for segmented predictions.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Lightning trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The Lightning module being trained or evaluated.
        stage : str
            The stage of the training process ('fit', 'validate', 'test', or 'predict').
        """
        if trainer.is_global_zero:
            print(f'Begin setup... for stage {stage}')
            print(f'Output directory: {self.p_outdir}')
            if self.p_tmp_outdir.exists():
                print(f'Removing existing temporary files in {self.p_tmp_outdir}...')
                for p_tmp in self.p_tmp_outdir.glob('*'):
                    p_tmp.unlink()
            else:
                print(f'Creating temporary directory... {self.p_tmp_outdir}')
                self.p_tmp_outdir.mkdir(parents=True, exist_ok=True)
            print('Setup done.')
        trainer.strategy.barrier()

    def write_on_batch_end(self, trainer, pl_module, prediction: list[dict], batch_indices, batch, batch_idx, dataloader_idx):
        """
        Store predictions from the current batch and save checkpointed results periodically.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Lightning trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The Lightning module being trained or evaluated.
        prediction : list of dict
            List of predictions from the batch.
        batch_indices : list
            Indices of the samples in the batch.
        batch : dict
            The input batch data.
        batch_idx : int
            Index of the current batch.
        dataloader_idx : int
            Index of the dataloader.
        """
        for pred_output in prediction:
            qset_uuid = pred_output['qset_uuid']
            seg_idx = pred_output['seg_idx']
            num_segments = pred_output['num_segments']
            self.rank_seg_preds.append((qset_uuid, seg_idx, num_segments, pred_output))

        if batch_idx % 100 == 0:  # checkpointing
            self.rank_seg_preds = sorted(self.rank_seg_preds, key=lambda x: x[:-1])
            torch.save(self.rank_seg_preds, self.p_tmp_outdir / f'rank-{trainer.global_rank}.pt')

    def on_predict_epoch_end(self, trainer, pl_module):
        """
        Merge segmented predictions across ranks and save results.

        Parameters
        ----------
        trainer : lightning.pytorch.Trainer
            Lightning trainer instance.
        pl_module : lightning.pytorch.LightningModule
            The Lightning module being trained or evaluated.
        """
        self.rank_seg_preds = sorted(self.rank_seg_preds, key=lambda x: x[:-1])
        torch.save(self.rank_seg_preds, self.p_tmp_outdir / f'rank-{trainer.global_rank}.pt')
        if trainer.world_size > 1:
            trainer.strategy.barrier()

        if trainer.is_global_zero:
            # get segmented features
            print('Gathering segmented features...')
            all_seg_preds = {}
            for p_pt in self.p_tmp_outdir.glob('*.pt'):
                rank_seg_preds = torch.load(p_pt, weights_only=True)
                for qset_uuid, seg_idx, num_segments, pred_output in rank_seg_preds:
                    if qset_uuid not in all_seg_preds:
                        all_seg_preds[qset_uuid] = [None] * num_segments
                    all_seg_preds[qset_uuid][seg_idx] = pred_output

            # merge features
            print('Merging features...')
            qset_preds = {}
            for qset_uuid, qset_seg_preds in all_seg_preds.items():
                new_ret_bboxes, new_ret_scores, frame_idxs = [], [], []
                num_segments = len(qset_seg_preds)
                for seg_idx, seg_pred in enumerate(qset_seg_preds):
                    assert seg_pred is not None, f'{qset_uuid}_{seg_idx}_{num_segments}'
                    new_ret_bboxes.append(seg_pred['ret_bboxes'])
                    new_ret_scores.append(seg_pred['ret_scores'])
                    frame_idxs.append(seg_pred['frame_idxs'])
                frame_idxs = torch.cat(frame_idxs, dim=0)
                mask_duplicated = frame_idxs == torch.cat([torch.tensor([-1]), frame_idxs[:-1]])
                qset_preds[qset_uuid] = {
                    'ret_bboxes': torch.cat(new_ret_bboxes, dim=0)[~mask_duplicated],
                    'ret_scores': torch.cat(new_ret_scores, dim=0)[~mask_duplicated],
                }

            # save intermediate results
            print('Saving intermediate results...')
            torch.save(qset_preds, self.p_int_pred)

            # TODO: Below should be handled by a separate evaluation script

            # get final predictions
            final_preds = get_final_preds_vq2d(qset_preds, split=self.split, movement=self.movement)

            if self.test_submit:
                # fix the order of the predictions
                final_preds = fix_predictions_order(
                    final_preds, self.official_anns_dir / f'vq_test_unannotated.json')

            # write the final predictions to json
            json.dump(final_preds, self.p_pred.open('w'))

            if not self.test_submit:
                # print metrics
                anno = f'data/vq_v2_val_{self.movement}_anno.json' if self.movement != "" else 'data/vq_v2_val_anno.json'
                subset_metrics = get_metrics_vq2d(Path(anno), self.p_pred)
                metrics_msg = format_metrics_vq2d(subset_metrics)
                print(metrics_msg)
                self.p_metrics_log.write_text(metrics_msg + '\n')
                json.dump({k: v['metrics'] for k, v in subset_metrics.items()}, self.p_metrics.open('w'))

            # remove temporary files
            for p_tmp in self.p_tmp_outdir.glob('*'):
                p_tmp.unlink()
            self.p_tmp_outdir.rmdir()


class PerSegmentWriterLaSOT(BasePredictionWriter):
    def __init__(self,
        output_dir,
        official_anns_dir,
        test_submit = False,
    ):
        super().__init__(write_interval="batch")
        self.p_outdir = Path(output_dir)
        self.p_int_pred = self.p_outdir / 'intermediate_predictions.pt'
        self.p_pred = self.p_outdir / 'predictions.json'
        self.p_metrics = self.p_outdir / 'metrics.json'
        self.p_metrics_log = self.p_outdir / 'metrics.log'
        self.rank_seg_preds = []
        self.test_submit = test_submit
        self.official_anns_dir = Path(official_anns_dir)

        self.split = 'test'

        self.p_tmp_outdir = self.p_outdir / 'tmp' / self.split

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        """Setup temporary directories for per-rank segmented predictions."""
        if trainer.is_global_zero:
            print(f'Begin setup... for stage {stage}')
            print(f'Output directory: {self.p_outdir}')
            if self.p_tmp_outdir.exists():
                print(f'Removing existing temporary files in {self.p_tmp_outdir}...')
                for p_tmp in self.p_tmp_outdir.glob('*'):
                    p_tmp.unlink()
            else:
                print(f'Creating temporary directory... {self.p_tmp_outdir}')
                self.p_tmp_outdir.mkdir(parents=True, exist_ok=True)
            print('Setup done.')
        trainer.strategy.barrier()

    def write_on_batch_end(self, trainer, pl_module, prediction: list[dict], batch_indices, batch, batch_idx, dataloader_idx):
        for pred_output in prediction:
            qset_uuid = pred_output['qset_uuid']
            seg_idx = pred_output['seg_idx']
            num_segments = pred_output['num_segments']
            self.rank_seg_preds.append((qset_uuid, seg_idx, num_segments, pred_output))

        if batch_idx % 100 == 0:  # checkpointing
            self.rank_seg_preds = sorted(self.rank_seg_preds, key=lambda x: x[:-1])
            torch.save(self.rank_seg_preds, self.p_tmp_outdir / f'rank-{trainer.global_rank}.pt')

    def on_predict_epoch_end(self, trainer, pl_module):
        """Merge segmented features and write to json."""

        self.rank_seg_preds = sorted(self.rank_seg_preds, key=lambda x: x[:-1])
        torch.save(self.rank_seg_preds, self.p_tmp_outdir / f'rank-{trainer.global_rank}.pt')
        if trainer.world_size > 1:
            trainer.strategy.barrier()

        if trainer.is_global_zero:
            # get segmented features
            print('Gathering segmented features...')
            all_seg_preds = {}
            for p_pt in self.p_tmp_outdir.glob('*.pt'):
                rank_seg_preds = torch.load(p_pt, weights_only=True)
                for qset_uuid, seg_idx, num_segments, pred_output in rank_seg_preds:
                    if qset_uuid not in all_seg_preds:
                        all_seg_preds[qset_uuid] = [None] * num_segments
                    all_seg_preds[qset_uuid][seg_idx] = pred_output

            # merge features
            print('Merging features...')
            qset_preds = {}
            for qset_uuid, qset_seg_preds in all_seg_preds.items():
                new_ret_bboxes, new_ret_scores, frame_idxs = [], [], []
                num_segments = len(qset_seg_preds)
                for seg_idx, seg_pred in enumerate(qset_seg_preds):
                    assert seg_pred is not None, f'{qset_uuid}_{seg_idx}_{num_segments}'
                    new_ret_bboxes.append(seg_pred['ret_bboxes'])
                    new_ret_scores.append(seg_pred['ret_scores'])
                    frame_idxs.append(seg_pred['frame_idxs'])
                frame_idxs = torch.cat(frame_idxs, dim=0)
                mask_duplicated = frame_idxs == torch.cat([torch.tensor([-1]), frame_idxs[:-1]])
                qset_preds[qset_uuid] = {
                    'ret_bboxes': torch.cat(new_ret_bboxes, dim=0)[~mask_duplicated],
                    'ret_scores': torch.cat(new_ret_scores, dim=0)[~mask_duplicated],
                }

            # save intermediate results
            print('Saving intermediate results...')
            torch.save(qset_preds, self.p_int_pred)

            # remove temporary files
            for p_tmp in self.p_tmp_outdir.glob('*'):
                p_tmp.unlink()
            self.p_tmp_outdir.rmdir()

            # # TODO: Below should be handled by a separate evaluation script

            # NOTE: no final results required for LaSOT

            p_clips_dir = Path("/data/datasets/LaSOT")
            metrics = get_metrics_lasot(p_clips_dir, self.p_int_pred)
            metrics_msg = format_metrics_lasot(metrics)
            print(metrics_msg)
            self.p_metrics_log.write_text(metrics_msg + '\n')
            json.dump(metrics, self.p_metrics.open('w'))
