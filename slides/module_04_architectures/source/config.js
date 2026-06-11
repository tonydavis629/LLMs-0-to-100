(function () {
  // Module 4 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 4',
    manimSections: {
      'bpe-training': [
        'BPETrainingScene_0000_start.mp4',
        'BPETrainingScene_0001_count_pairs.mp4',
        'BPETrainingScene_0002_merge_lo.mp4',
        'BPETrainingScene_0003_merge_low.mp4',
        'BPETrainingScene_0004_result.mp4'
      ],
      'recurrence-vs-attention': [
        'RecurrenceVsAttentionScene_0000_recurrence.mp4',
        'RecurrenceVsAttentionScene_0001_attention.mp4'
      ],
      'sampling-demo': [
        'SamplingScene_0000_dist.mp4',
        'SamplingScene_0001_temp_sharp.mp4',
        'SamplingScene_0002_temp_flat.mp4',
        'SamplingScene_0003_topk.mp4',
        'SamplingScene_0004_topp.mp4'
      ],
      'embedding-lookup': [
        'EmbeddingLookupScene_0000_word.mp4',
        'EmbeddingLookupScene_0001_lookup.mp4',
        'EmbeddingLookupScene_0002_vector.mp4'
      ],
      'ffn-expand': [
        'FFNExpandScene_0000_vector.mp4',
        'FFNExpandScene_0001_expand.mp4',
        'FFNExpandScene_0002_activate.mp4',
        'FFNExpandScene_0003_contract.mp4'
      ],
      'norm-demo': [
        'NormDemoScene_0000_vector.mp4',
        'NormDemoScene_0001_layernorm.mp4',
        'NormDemoScene_0002_rmsnorm.mp4'
      ],
      'residual-stream': [
        'ResidualStreamScene_0000_stream.mp4',
        'ResidualStreamScene_0001_block1.mp4',
        'ResidualStreamScene_0002_block2.mp4',
        'ResidualStreamScene_0003_readout.mp4'
      ]
    },
    widgets: {}
  };
}());
