
Differences from MLP-Mixer:
- No batch norm.
- Does not operate on tokens (derived from image patches), but on the input image/array directly*.
    (*Or a representation of the input array created using Mixers (maybe using linear MLPs), to resize, see implementation.)
- Keeps same dimension as input.
- Does skip connections differently (only after a Mixer layer, no longer during).
- Initially the model is the identity function (leaving out the classifier part).
- Different classifier method, with no averaging.

Todo: 
- Cluster & compare similarity of hidden representations
- Rotations
- Position encodings
