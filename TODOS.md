# When enabling real-time computation for a given sequence of points, some debt was left and the previous code is broken.

- [ ] In model, the functions were made for batch computation and were not properly adapted.
- [ ] Jolib removed from (code left broken when it's not used):
    - 'baseline_utils' 
    - 'baseline utils' 
- [ ] Argoverse dependency removed from 'baseline_utils' with viz code left broken.
- [ ] In LSTMDataset, 'candidate_centerlines', 'candidate_nt_distances' and 'seq_paths' were all disabled.
- [ ] In ModelUtils, by default now the prefix module. is removed from the state dict (when using joblib this needs to be fixed).
