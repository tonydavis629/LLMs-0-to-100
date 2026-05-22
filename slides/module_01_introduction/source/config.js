// Module 1 - Introduction: per-module config consumed by the shared slides.js.
window.MODULE_CONFIG = {
  title: 'LLMs 0 to 100 - Module 1',
  // This module has no manim step-through videos.
  manimSections: {},
  // AI Summers and Winters chart, drawn the first time its slide appears.
  onSlideChanged: function(event) {
      if (event.currentSlide.id === 'summers-winters') {
        var canvas = document.getElementById('hypeChart');
        if (canvas && !canvas._chartCreated) {
          canvas._chartCreated = true;
          // Custom plugin to draw labeled vertical lines (more reliable than annotation plugin on category axes)
          var eventLinesPlugin = {
            id: 'eventLines',
            afterDraw: function(chart) {
              var ctx = chart.ctx;
              var xScale = chart.scales.x;
              var yScale = chart.scales.y;
              var events = [
                { label: 'Dartmouth\nWorkshop', year: '1956', color: '#f5a623' },
                { label: 'Lighthill\nReport', year: '1974', color: '#6cb4ff' },
                { label: 'Expert\nSystems', year: '1984', color: '#f5a623' },
                { label: '2nd AI\nWinter', year: '1993', color: '#6cb4ff' },
                { label: 'AlexNet', year: '2012', color: '#50c878' },
                { label: 'Transformers', year: '2017', color: '#f5a623' },
                { label: 'LLM Era', year: '2024', color: '#f5a623' }
              ];
              events.forEach(function(ev) {
                var labelIdx = chart.data.labels.indexOf(ev.year);
                if (labelIdx === -1) return;
                var x = xScale.getPixelForValue(labelIdx);
                var yTop = yScale.top;
                var yBottom = yScale.bottom;
                // Dashed vertical line
                ctx.save();
                ctx.setLineDash([4, 4]);
                ctx.strokeStyle = ev.color;
                ctx.globalAlpha = 0.5;
                ctx.lineWidth = 1;
                ctx.beginPath();
                ctx.moveTo(x, yTop);
                ctx.lineTo(x, yBottom);
                ctx.stroke();
                ctx.restore();
                // Label at top
                ctx.save();
                ctx.font = '10px Inter, sans-serif';
                ctx.fillStyle = ev.color;
                ctx.textAlign = 'center';
                var lines = ev.label.split('\n');
                for (var i = 0; i < lines.length; i++) {
                  ctx.fillText(lines[i], x, yTop - (lines.length - i) * 12 + 10);
                }
                ctx.restore();
              });
            }
          };
          new Chart(canvas.getContext('2d'), {
            type: 'line',
            plugins: [eventLinesPlugin],
            data: {
              labels: [
                '1950', '1953', '1956', '1958', '1960', '1963', '1965', '1968',
                '1970', '1972', '1974', '1976', '1978', '1980', '1982', '1984',
                '1986', '1988', '1990', '1993', '1996', '2000', '2004', '2008',
                '2012', '2014', '2016', '2017', '2018', '2019', '2020', '2021',
                '2022', '2023', '2024'
              ],
              datasets: [{
                label: 'AI Hype / Investment',
                data: [
                  10, 30, 75, 70, 65, 55, 50, 35,
                  25, 18, 10, 8, 10, 35, 50, 65,
                  70, 50, 35, 18, 15, 20, 22, 28,
                  50, 55, 65, 75, 85, 95, 110, 140,
                  200, 320, 500
                ],
                borderColor: '#4a9eff',
                backgroundColor: 'rgba(74, 158, 255, 0.08)',
                fill: true,
                tension: 0.4,
                pointRadius: 0,
                borderWidth: 2.5
              }]
            },
            options: {
              responsive: true,
              maintainAspectRatio: false,
              animation: { duration: 1200 },
              layout: { padding: { top: 30 } },
              scales: {
                x: {
                  title: { display: true, text: 'Year', color: '#8892a4', font: { size: 13 } },
                  ticks: { color: '#8892a4', maxTicksLimit: 12, font: { size: 11 } },
                  grid: { color: 'rgba(42, 52, 80, 0.5)' }
                },
                y: {
                  title: { display: true, text: 'Hype / Investment', color: '#8892a4', font: { size: 13 } },
                  ticks: { display: false },
                  grid: { color: 'rgba(42, 52, 80, 0.5)' },
                  min: 0,
                  max: 550
                }
              },
              plugins: {
                legend: { display: false }
              }
            }
          });
        }
      }
  }
};
