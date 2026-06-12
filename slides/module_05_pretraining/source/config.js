(function () {
  // Module 5 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 5',
    manimSections: {
      'next-token': [
        'NextTokenScene_0000_sequence.mp4',
        'NextTokenScene_0001_shift.mp4',
        'NextTokenScene_0002_predict.mp4',
        'NextTokenScene_0003_target.mp4',
        'NextTokenScene_0004_loss.mp4'
      ],
      'training-loop': [
        'TrainingLoopScene_0000_setup.mp4',
        'TrainingLoopScene_0001_forward.mp4',
        'TrainingLoopScene_0002_backward.mp4',
        'TrainingLoopScene_0003_update.mp4',
        'TrainingLoopScene_0004_descend.mp4'
      ],
      'sequence-packing': [
        'SequencePackingScene_0000_docs.mp4',
        'SequencePackingScene_0001_concat.mp4',
        'SequencePackingScene_0002_chop.mp4',
        'SequencePackingScene_0003_batch.mp4'
      ],
      'lr-schedule': [
        'LRScheduleScene_0000_axes.mp4',
        'LRScheduleScene_0001_warmup.mp4',
        'LRScheduleScene_0002_cosine.mp4',
        'LRScheduleScene_0003_annotate.mp4'
      ],
      'scaling-laws': [
        'ScalingLawScene_0000_powerlaw.mp4',
        'ScalingLawScene_0001_extrapolate.mp4',
        'ScalingLawScene_0002_chinchilla.mp4',
        'ScalingLawScene_0003_rule.mp4'
      ],
      'data-parallel': [
        'DataParallelScene_0000_replicas.mp4',
        'DataParallelScene_0001_split.mp4',
        'DataParallelScene_0002_localgrad.mp4',
        'DataParallelScene_0003_allreduce.mp4'
      ],
      'tensor-parallel': [
        'TensorParallelScene_0000_matmul.mp4',
        'TensorParallelScene_0001_split.mp4',
        'TensorParallelScene_0002_partial.mp4',
        'TensorParallelScene_0003_gather.mp4'
      ],
      'fsdp': [
        'FSDPScene_0000_copies.mp4',
        'FSDPScene_0001_shard.mp4',
        'FSDPScene_0002_batch.mp4',
        'FSDPScene_0003_gather.mp4',
        'FSDPScene_0004_free.mp4'
      ],
      'perplexity': [
        'PerplexityScene_0000_spread.mp4',
        'PerplexityScene_0001_perplexity.mp4',
        'PerplexityScene_0002_sharpen.mp4',
        'PerplexityScene_0003_bits.mp4'
      ]
    },
    widgets: {}
  };
}());
