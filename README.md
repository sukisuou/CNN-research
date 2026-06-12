##### ================================== Experimental Log ==================================
##### 1.0 : Basic CNN + ANN                             - 0.9877, 0.0390     --> [acc, loss]
##### 1.1 : added bottlenecking                         - 0.9890, 0.0384
##### 1.2 : C2f style with bottleneck                   - 0.9883, 0.0339
##### 2.0 : full CNN model                              - 0.9838, 0.0522
##### 2.1 : half channels                               - 0.9803, 0.0647
##### 2.2 : learnt CNN classifier head with conv layer
##### 2.3 : introduces strides                          - 0.9737, 0.0878, no early_stopping
##### 2.4 : too aggressive, use maxpool (ref #2.0)      - 0.9885, 0.0351
##### 3.0 : taking inspiration from YOLO's C2f block
##### 3.1 : introduces splitting (32, 32) -> (32, 16)   - 0.9842, 0.0495, no early_stopping
##### 3.2 : no dimension reduction (1x1 projection)     - 0.9861, 0.0454, no early_stopping
##### deduction : optimization diff increases as model becomes more sophisticated
##### 4.0 : increases epoch (50 -> 100) for no-early-stop cases
##### 4.1 : 1x1 project rerun                           - 0.9876, 0.0418 [67]
##### ======================================================================================