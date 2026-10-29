(function () {
  // Module 8 config. Manim steppers are keyed by the `scene` slug used in the
  // :::manim fences; each maps to its ordered list of section clips.
  window.MODULE_CONFIG = {
    title: 'LLMs 0 to 100 - Module 8',
    manimSections: {
      'patchify': [
        'PatchifyScene_0000_image.mp4',
        'PatchifyScene_0001_grid.mp4',
        'PatchifyScene_0002_flatten.mp4',
        'PatchifyScene_0003_project.mp4'
      ],
      'videobudget': [
        'VideoBudgetScene_0000_frame.mp4',
        'VideoBudgetScene_0001_frames.mp4',
        'VideoBudgetScene_0002_budget.mp4',
        'VideoBudgetScene_0003_overflow.mp4'
      ],
      'posembed': [
        'PosEmbedScene_0000_grid.mp4',
        'PosEmbedScene_0001_flatten.mp4',
        'PosEmbedScene_0002_add.mp4',
        'PosEmbedScene_0003_similarity.mp4'
      ],
      'audiotokens': [
        'AudioTokensScene_0000_wave.mp4',
        'AudioTokensScene_0001_spectrogram.mp4',
        'AudioTokensScene_0002_encoder.mp4',
        'AudioTokensScene_0003_codec.mp4',
        'AudioTokensScene_0004_sequence.mp4'
      ]
    },
    widgets: {}
  };
}());
