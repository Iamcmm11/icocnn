# DCASE2025 Task3 Stereo SELD Dataset

## Authors and affiliations

[Sony AI and Sony Group Corporation](https://sony.github.io/creativeai/)

[Audio Research Group / Tampere University](https://webpages.tuni.fi/arg/)

[Queen Mary University of London](https://www.qmul.ac.uk/)

**Sony**
- Kazuki Shimada ([contact](mailto:kazuki.shimada@sony.com))
- Kengo Uchida
- Yuichiro Koyama
- Naoya Takahashi
- Takashi Shibuya
- Shusuke Takahashi 
- Yuki Mitsufuji ([contact](mailto:yuhki.mitsufuji@sony.com))

**Tampere University**
- Archontis Politis ([contact](mailto:archontis.politis@tuni.fi))
- Parthasaarathy Sudarsanam
- David Díaz-Guerra
- Ruchi Pandey
- Tuomas Virtanen ([contact](mailto:tuomas.virtanen@tuni.fi))

**Queen Mary University of London**
- Iran Roman ([contact](mailti:i.roman@qmul.ac.uk))



## Description

The **DCASE2025 Task3 Stereo SELD Dataset** is a stereo audio and video dataset derived from the STARSS23 dataset. The original STARSS23's first-order Ambisonics (FOA) audio and 360° video data have been converted to **stereo audio** and **perspective video** data, **simulating regular media content**. These clips serve as the development and evaluation dataset for the [sound event localization and detection (SELD) task](https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content) of the [DCASE2025 Challenge](https://dcase.community/challenge2025/).

The STARSS23 dataset contains multichannel recordings of sound scenes in various rooms and environments, together with temporal and spatial annotations of prominent events belonging to a set of target classes. The 360° video are spatially and temporally aligned with the microphone array recordings.

To construct the **DCASE2025 Task3 Stereo SELD Dataset**, we conduct the following sampling and conversion procedures from the STARSS23 dataset. We first sample 5-second clips from the original STARSS23 recordings. Then, we convert the 5-second FOA audio and 360° video to generate stereo audio and perspective video data corresponding to a fixed point-of-view. According to the fixed viewing angle, we first rotate the FOA audio. Then, we convert the rotated FOA audio to stereo audio, emulating a mid-side (M/S) recording technique. We convert the equirectangular video to a perspective video with the same viewing angle as the audio. We set the horizontal field-of-view (FOV) to 100 degrees and the video resolution (Width:Height) to 640:360 pixels, with an aspect ratio of 16:9, which is widely used in media content.

We also rotate the original STARSS23's direction-of-arrival (DOA) labels to new DOA labels centered at the fixed viewing angle. The new azimuth labels are folded back from back to front, considering front-back ambiguity. The elevation labels are omitted due to top-bottom ambiguity. The distance labels are kept the same as the STARSS23 one. To get the binary onscreen/offscreen event labels, we compare the new DOA labels with the FOV in the perspective video.

Please check the [challenge webpage](https://dcase.community/challenge2025/task-stereo-sound-event-localization-and-detection-in-regular-video-content) for details missing in this description.



## Report and reference

A technical report about this dataset will be published.



## Aim
The DCASE2025 Task3 Stereo SELD Dataset is suitable for training and evaluation of machine-listening models for sound event detection (SED), general sound source localization with diverse sounds or signal-of-interest localization, and joint sound event localization and detection (SELD). Additionally, the dataset can be used to evaluate signal processing methods that do not necessarily rely on training, such as acoustic source localization methods and multiple-source acoustic tracking. The dataset allows evaluation of the performance and robustness of the aforementioned applications for diverse types of sounds and under diverse acoustic conditions.

Specifically, the DCASE2025 Task3 Stereo SELD Dataset allows us to evaluate models using stereo audio data and explore tasks in common audio and media scenarios.



## Specifications

The specifications of the stereo audio and video dataset can be summarized in the following:

Recording (STARSS22/23 setup):
- Each recording clip is part of a recording session happening in a unique room.
- Groups of participants, sound making props, and scene scenarios are unique for each session (with a few exceptions).
- 13 target classes are identified in the recordings and strongly annotated by humans.
- Spatial annotations for those active events are captured by an optical tracking system.
- Sound events out of the target classes are considered as interference.
- Occurrences of up to 3 simultaneous events are fairly common, while higher numbers of overlapping events (up to 6) can occur but are rare.

Sampling and conversion:
- Each sampling step randomly selects its recording, start frame, and viewing angle.
- A recording is selected with length-weighted random choice to treat all frames of all files equally.
- A start frame is selected uniformly within each recording.
- A horizontal viewing angle is selected uniformly at 360° while the vertical viewing angle is kept at 0° elevation.
- 12 audio recordings with missing videos (fold3_room21_mix001.wav - fold3_room21_mix012.wav) are not selected to keep the same set between audio-only and audiovisual tracks.
- Several clips do not contain any target sound events after random sampling.
- The class distribution across all frames after random sampling is similar to the STARSS23 one.
- The onscreen/offscreen distribution across all frames is around 1 : 3.

Volume, duration, and data split:
- A total of 16 unique rooms were captured in the recordings (development set).
- 30,000 clips of 5-sec duration, with a total time of 41.7 hrs (development dataset).
- 23.9 % of the clips are derived from recordings in Tokyo (development dataset).
- 76.1 % of the clips are derived from recordings in Tampere (development dataset).
- 10,000 clips of 5-sec duration, with a total time of 13.9 hrs, derived from recordings in both sites (evaluation dataset).
- A training-testing split is provided for reporting results using the development dataset.
- 2 rooms in Tokyo are for the training split (dev-train-sony).
- 2 rooms in Tokyo are for the testing split (dev-test-sony).
- 7 rooms in Tampere are for the training split (dev-train-tau).
- 5 rooms in Tampere are for the testing split (dev-test-tau).

Audio:
- Sampling rate: 24kHz.
- Bit depth:     16 bits.
- Stereo format: mid-side (M/S) technique with left-right cardioid stereo patterns.

Video:
- Video format:                  perspective.
- Video resolution:              640x360.
- Video frames per second (fps): 29.97.



## Sound event classes

13 target sound event classes were annotated. The classes follow loosely the [AudioSet ontology](https://research.google.com/audioset/ontology/index.html).

0. Female speech, woman speaking
1. Male speech, man speaking
2. Clapping
3. Telephone
4. Laughter
5. Domestic sounds
6. Walk, footsteps
7. Door, open or close
8. Music
9. Musical instrument
10. Water tap, faucet
11. Bell
12. Knock



## Naming convention

The audio files in the development dataset follow the naming convention:
- fold[fold number]_room[room number]_mix[recording number per room]_deg[viewing angle in degree]_start[start time in frame].wav

The fold number at the moment is used only to distinguish between the training and testing split. The room information is provided for the dataset user to potentially help understand the performance of their method concerning different conditions.

Each clip is generated by randomly selecting the recording, viewing angle, and start time. The recording number, viewing angle, and start time are provided to indicate the configuration of the clip. Note that the viewing angle and start time are not sampled at equal intervals but sampled randomly.

The video and metadata files have the same folder structure and naming convention as the audio files.

The audio files in the evaluation dataset consists of clips without any information on the origin or on the location in the naming convention, as below:
- sample[clip number].wav



## Example application

An implementation of a trainable model performing joint SELD, trained and evaluated with this dataset, is provided [here](https://github.com/partha2409/DCASE2025_seld_baseline). This implementation will serve as the baseline method in Task 3 of the DCASE2025 Challenge under audio-only and audiovisual inference tracks.

The stereo SELD data generator for this dataset is also available [here](https://github.com/SonyResearch/dcase2025_stereo_seld_data_generator). The data generator can construct stereo SELD datasets like the DCASE2025 Task3 Stereo SELD Dataset from real or synthetic FOA SELD datasets. The generator samples a clip randomly and converts its FOA audio / 360° video / metadata to new stereo audio / perspective video / metadata according to a viewing angle.



## Development and evaluation

The current version (Version 1.1.0) of the dataset includes development audio/video clips and labels and the evaluation clips without labels, used by the participants of Task 3 of the DCASE2025 Challenge to train and validate their submitted systems (development), and produce system outputs for the challenge evaluation phase.

If researchers wish to compare their system against the submissions of DCASE2025 Challenge, they will have directly comparable results if they use the evaluation data as their testing set.



## Download instruction

The file `stereo_dev.zip` corresponds to stereo audio data for the development dataset.
The file `video_dev.zip` contains the perspective videos for the development dataset.
The file `metadata_dev.zip` contains the metadata for the development dataset.

The file `stereo_eval.zip` corresponds to stereo audio data for the evaluation dataset.
The file `video_eval.zip` contains the perspective videos for the evaluation dataset.

Download the zip files and use your favorite compression tool to unzip these zip files.



## License

This dataset is licensed under the [MIT](https://opensource.org/licenses/MIT) license.
