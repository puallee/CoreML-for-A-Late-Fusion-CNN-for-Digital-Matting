# CoreML-for-A-Late-Fusion-CNN-for-Digital-Matting
====
<BR/>This is the github project: CoreML-for-A-Late-Fusion-CNN-for-Digital-Matting 
<BR/>We obtain a CoreML model from A-Late-Fusion-CNN-for-Digital-Matting https://github.com/yunkezhang/FusionMatting in IOS 12
<BR/> we rewrite tensorflow function to keras function
<BR/> Because of the limitation of coreml, coreml does not support keras'sLambda: In fusion_blocks.py, we do factorization for(1-bg_out)(1-fg_weights)=bg_out*fg_weights--bg_out-fg_weights+1 to replace Lambda.
<BR/>Ubuntu 16.04 gcc 5.4 Python 3.6  TensorFlow 2.2.0 + Keras 2.3.1 
<BR/>coremltools 3.4 in MAC 10.4
<BR/> if you need some CoreML model for your IOS, please contact me.









































