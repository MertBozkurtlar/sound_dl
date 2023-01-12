# Von-Mises Network for Sound Source Localization

File Structure:\
&nbsp;&nbsp;main.py: Includes the main pipeline script for running the program\
&nbsp;&nbsp;modules:\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-constants.py: Includes settings for running the program\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-dataset.py: Includes the data processing scripts and Dataset class\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-model.py: Includes the Neural Network and Von-Mises Layer\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-train.py: Includes the scripts for training the model\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-input.py: Includes the scripts for getting input from microphone array\
&nbsp;&nbsp;model: Model state save location


Data Folder Structure:\
&nbsp;&nbsp;Folder structure for data assumed to be in following structure:\
&nbsp;&nbsp;Data/Noise_Type/SNR/(sp-deg_$Angle.wav)

&nbsp;&nbsp;Data: The main folder\
&nbsp;&nbsp;Noise_Type: The type of the noise added to recordings\
&nbsp;&nbsp;SNR: Signal to noise ratio of the noise added to recordings\
&nbsp;&nbsp;sp-deg_$Angle.wav: Audio data of recording coming from $Angle

&nbsp;&nbsp;Example: Data/WHN/20/sp-deg_005.wav
