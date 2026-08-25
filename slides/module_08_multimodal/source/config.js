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
      ]
    },
    widgets: {}
  };
}());
