# Measurement vs ground truth: loss_focal_tversky

85 test images, threshold 0.96, model input 320x320. All values in pixels of the original image.
GT mean crack width: median 5.92 px, range 1.15-38.87 px; 17 images thin (< 3 px), 68 wider.

Signed error = measured - GT (positive = reads too wide / too long / too many).
CEILING = perfect model at this input size; error it shares with MODEL comes from the resize.

## All images (n=85)

| quantity | source | median signed | median abs | within 1 px | median rel % |
|---|---|---|---|---|---|
| mean_width | model | +0.36 | 0.98 | 51% | 18% |
| mean_width | ceiling | +0.10 | 0.10 | 99% | 2% |
| p95_width | model | +0.00 | 1.86 | 36% | 16% |
| p95_width | ceiling | +0.00 | 0.00 | 96% | 0% |
| max_width | model | +0.00 | 2.00 | 27% | 14% |
| max_width | ceiling | +0.00 | 0.00 | 99% | 0% |
| total_length | model | -14.00 | 75.00 | 6% | 10% |
| total_length | ceiling | -10.00 | 10.00 | 22% | 2% |
| n_components | model | +0.00 | 1.00 | 62% | 30% |
| n_components | ceiling | +0.00 | 0.00 | 89% | 0% |

## Thin cracks (GT mean width < 3 px) (n=17)

| quantity | source | median signed | median abs | within 1 px | median rel % |
|---|---|---|---|---|---|
| mean_width | model | +0.97 | 0.97 | 53% | 48% |
| mean_width | ceiling | +0.07 | 0.07 | 100% | 3% |
| p95_width | model | +1.53 | 1.53 | 47% | 40% |
| p95_width | ceiling | +0.00 | 0.00 | 94% | 0% |
| max_width | model | +1.53 | 2.94 | 6% | 44% |
| max_width | ceiling | +0.00 | 0.00 | 94% | 0% |
| total_length | model | -58.00 | 70.00 | 18% | 12% |
| total_length | ceiling | -5.00 | 5.00 | 18% | 1% |
| n_components | model | +0.00 | 3.00 | 29% | 55% |
| n_components | ceiling | +0.00 | 0.00 | 71% | 0% |

## Wider cracks (>= 3 px) (n=68)

| quantity | source | median signed | median abs | within 1 px | median rel % |
|---|---|---|---|---|---|
| mean_width | model | +0.10 | 1.00 | 50% | 16% |
| mean_width | ceiling | +0.10 | 0.10 | 99% | 2% |
| p95_width | model | +0.00 | 1.88 | 34% | 15% |
| p95_width | ceiling | +0.00 | 0.03 | 97% | 0% |
| max_width | model | -0.13 | 2.00 | 32% | 12% |
| max_width | ceiling | +0.00 | 0.00 | 100% | 0% |
| total_length | model | -11.00 | 77.50 | 3% | 9% |
| total_length | ceiling | -11.50 | 11.50 | 24% | 2% |
| n_components | model | +0.00 | 1.00 | 71% | 27% |
| n_components | ceiling | +0.00 | 0.00 | 94% | 0% |
