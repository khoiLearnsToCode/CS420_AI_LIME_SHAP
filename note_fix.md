## 4.6 Feature importance is not a causal effect

"Four reasons, each visible in our experiment, keep the two apart." 
=> "Four reasons, each visible in our experiment, show why the two should not be confused."

"Unrealistic samples. Both methods change each feature independently (for KernelSHAP, the independence assumption of Section 2.3), creating combinations that never occur in real tumours, such as a large area with a small radius."
=> "Unrealistic samples. Both LIME and KernelSHAP change each feature independently of the others; for KernelSHAP this is the feature-independence assumption described in Section 2.3. This creates combinations that never occur in real tumours, such as a large area with a small radius, so the model is queried in regions where it has never seen data."