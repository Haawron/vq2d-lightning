from lightning.pytorch.callbacks import ModelCheckpoint


def main():
    model_checkpoints = [
        ModelCheckpoint(
            save_last=False,
            monitor='nlq/R1@0.3',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=1,
            filename='step={step}-nlq_R1@0.3={nlq/R1@0.3:.4f}'),
        ModelCheckpoint(
            save_last=False,
            monitor='nlq/R5@0.3',
            auto_insert_metric_name=False,
            mode='max',
            save_top_k=5,
            filename='step={step}-nlq_R5@0.3={nlq/R5@0.3:.4f}'),
    ]

    test_metrics_for_validating_filename_templates = {
        'nlq/R1@0.3': 23.51,
        'nlq/R5@0.3': 34.23,
    }
    print('\n' + '=' * 80 + '\n')
    print('Validate Model Checkpoint Filename Templates:')
    for model_checkpoint in model_checkpoints:
        print(model_checkpoint.format_checkpoint_name(test_metrics_for_validating_filename_templates))
    print('\n' + '=' * 80 + '\n')


if __name__ == '__main__':
    main()
