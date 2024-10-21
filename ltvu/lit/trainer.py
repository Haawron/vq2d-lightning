import json
from pathlib import Path
import datetime
import hydra.utils
from omegaconf import OmegaConf, DictConfig, open_dict

import torch

import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor, ModelSummary, ModelCheckpoint, TQDMProgressBar,
    BasePredictionWriter
)
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from ltvu.utils.compute_results import get_final_preds
from ltvu.metrics import get_metrics, format_metrics


type_loggers = WandbLogger | CSVLogger


class PerSegmentWriter(BasePredictionWriter):
    def __init__(self,
        output_dir,
        test_submit = False,
    ):
        super().__init__(write_interval="batch")
        self.p_tmp_outdir = Path(output_dir) / 'tmp'
        self.p_tmp_outdir.mkdir(parents=True, exist_ok=True)
        self.p_int_pred = Path(output_dir) / 'intermediate_predictions.pt'
        self.p_pred = Path(output_dir) / 'predictions.json'
        self.p_metrics = Path(output_dir) / 'metrics.json'
        self.p_metrics_log = Path(output_dir) / 'metrics.log'
        for p_tmp in self.p_tmp_outdir.glob('*'):
            p_tmp.unlink()
        self.rank_seg_preds = []
        self.test_submit = test_submit

        if self.test_submit:
            self.split = 'test_unannotated'
            self.p_pred = self.p_pred.with_name('test_predictions.json')
            self.p_int_pred = self.p_int_pred.with_name('test_intermediate_predictions.pt')
        else:
            self.split = 'val'

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
            print('Getting segmented features...')
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
            final_preds = get_final_preds(qset_preds, split=self.split)

            # write the final predictions to json
            json.dump(final_preds, self.p_pred.open('w'))

            if not self.test_submit:
                # print metrics
                subset_metrics = get_metrics(Path(f'data/vq_v2_val_anno.json'), self.p_pred)
                metrics_msg = format_metrics(subset_metrics)
                print(metrics_msg)
                self.p_metrics_log.write_text(metrics_msg + '\n')
                json.dump({k: v['metrics'] for k, v in subset_metrics.items()}, self.p_metrics.open('w'))

            # remove temporary files
            for p_tmp in self.p_tmp_outdir.glob('*'):
                p_tmp.unlink()
            self.p_tmp_outdir.rmdir()


def get_trainer(config, jid, enable_progress_bar=False, enable_checkpointing=True, ddp_timeout=30):
    runtime_outdir: str = config.runtime_outdir
    trainer_config: DictConfig = config.trainer

    # callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(),
        TQDMProgressBar(refresh_rate=1 if enable_progress_bar else 20, leave=True),
        PerSegmentWriter(
            output_dir=runtime_outdir,
            test_submit=config.dataset.get('test_submit', False),
        ),
    ]
    if enable_checkpointing:
        ckpt_callback_iou = ModelCheckpoint(
            dirpath=runtime_outdir,
            save_last=False,
            monitor='Val/iou',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='epoch={epoch}-iou={Val/iou:.4f}')
        ckpt_callback_prob = ModelCheckpoint(
            dirpath=runtime_outdir,
            save_last=False,
            monitor='Val/prob_acc',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='epoch={epoch}-prob_acc={Val/prob_acc:.4f}')
        callbacks.append(ckpt_callback_iou)
        callbacks.append(ckpt_callback_prob)
    else:
        ckpt_callback_prob = None

    assert jid is not None, 'jid must be provided when loggers are enabled'
    with open_dict(trainer_config):  # obtaining write access
        loggers_config = trainer_config.pop('logger', [])  # to not pass it to the Trainer

    loggers = [CSVLogger(save_dir=runtime_outdir, name="lit", version=jid)]
    for logger_config in loggers_config:
        logger: type_loggers = hydra.utils.instantiate(logger_config)
        loggers.append(logger)

    # Note: do not let hydra instantiate the Trainer or it is highly inflexible
    trainer_config = OmegaConf.to_container(trainer_config, resolve=True)
    if 'strategy' not in trainer_config:
        trainer_config['strategy'] = DDPStrategy(
            timeout=datetime.timedelta(ddp_timeout),
            find_unused_parameters=True)
    trainer = L.Trainer(
        **trainer_config,
        enable_model_summary=False,
        default_root_dir=runtime_outdir,
        logger=loggers,
        callbacks=callbacks,
    )
    return trainer, ckpt_callback_prob
