## Instructions for downloading each of the datasets

### ImageNet, RedPajama, MNIST

These three downloads are managed through [HuggingFace](https://huggingface.co/). You will need to create an account and provide credentials by adding your HuggingFace token to the environment before running any download. In particular, these two environment variables are required:
1. `HF_TOKEN`: Set this to the User Access Token in your HuggingFace profile
2. `HF_HOME`: Set this to the directory you would like HuggingFace downloads to be stored

See more information on environment setup at [HuggingFace's official documentation](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables).

### Note on `ffprobe` for Computer Vision datasets

We use ffprobe to read the duration of each video in our video dataloaders. While opencv-python does not require a library installation external to Python, we found opencv to be more unreliable than ffprobe at reading video durations.

#### ffprobe setup: 
On a linux system, you can `apt install ffmpeg` to get ffprobe. If ffprobe is in your path, the training scripts should run without any additional setup for ffprobe.

Or if you have a conda env you can do `conda install -c conda-forge ffmpeg` and see if that works.

If none of those work, you can download ffprobe at the [ffmpeg download site](https://ffmpeg.org//download.html). Once downloaded, extract the binaries and provide the path to the ffprobe file by either setting:
- the environment variable `FFPROBE_PATH=<path_to_ffprobe>`
- or the command-line argument `--ffprobe_path=<path_to_ffprobe>`

### Kinetics-400 & Kinetics-600

To download Kinetics-400 and Kinetics-600, use the scripts at https://github.com/cvdfoundation/kinetics-dataset. Set the command-line argument `--dataset_dir=<path_to_dataset>`. Then set the `$K400_DIR` env variable.

### Something-something-v2
SSv2 is distributed by Qualcomm in separate files at [this link](https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads). Download the video files and the labels. After downloading all the video files, you can concatenate them and unzip them as a single file:

```
cat 20bn-something-something-v2-* > ssv2_archive
tar -xvf ssv2_archive
```
Also unzip the labels: ``unzip 20bn-something-something-download-package-labels.zip``
Then, set the command-line argument `--dataset_dir=<path_to_dataset>`, or preferably set the `$SSV2_DIR` env variable.

### UCF-101

Download the UCF dataset and annotations from UCF's website: https://www.crcv.ucf.edu/data/UCF101.php. *Note: as www.crcv.ucf.edu does not have a trusted certificate, if using wget to download you must provide the argument `--no-check-certificate`.*

The UCF dataset directory structure should appear as:
- UCF101
    - ucfTrainTestlist/ (UCF dataset annotations)
    - *.avi (all UCF dataset videos)

Set the command-line argument `--dataset_dir=<path_to_dataset>`. Or, based on the current scripts, you can set the env variable $SSV2_DIR