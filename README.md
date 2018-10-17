
Nvidia-driver, CUDA, CUDNN:
Driver:
Already installed, nvidia-driver 384.130. 
**Note: ** Shut down the laptop first then close the cover. It may lead to an error of cuda if the cover is closed without shutting down the laptop.
CUDA:
Check:
```
cd /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery
./deviceQuery
```
The result should be “PASS”. Then,
```
cd /usr/local/cuda-9.0/samples/1_Utilities/bandwidthTest
./bandwidthTest
```
The result should be “PASS”.
If any result was “FAIL”, delete CUDA and reinstalled.
Uninstall: ```cd /usr/local/cuda-9.0/bin/``` use file ```uninstall_cuda_9.0.pl```
Reinstall: Use the runfile ```Downloads/cuda_9.0.176_384.81_linux.run``` to install it. CUDA must be installed by runfile, or it will lead to a TPM error.
In installation, choose:
```
accept  # For accept installation
n  # Do not install driver
y  # install cuda toolkit
<Enter>  # install to default dir
y # softlink
n  # Do not copy samples
```
# hello-world

hello github!
First time to use it. Hope I can upload much in the future.
