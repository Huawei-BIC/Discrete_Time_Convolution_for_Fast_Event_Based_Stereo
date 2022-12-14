#################################################################
This code is a demo of CVPR 2022 paper "Discrete time convolution for 
fast event-based stereo".

Our code is developed based on the code from ICCV2019 paper "Learning an event sequence embedding 
for dense event-based deep stereo"

paper: https://openaccess.thecvf.com/content_ICCV_2019/papers/Tulyakov_Learning_an_Event_Sequence_Embedding_for_Dense_Event-Based_Deep_Stereo_ICCV_2019_paper.pdf
code: https://github.com/tlkvstepan/event_stereo_ICCV2019

@inproceedings{tulyakov-et-al-2019,
  author = {Tulyakov, S. and Fleuret, F. and Kiefel, M. and Gehler, P. and Hirsch, M.},
  title = {Learning an event sequence embedding for event-based deep stereo},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  year = {2019},
  type = {Oral},
  note = {To appear},
  url = {https://fleuret.org/papers/tulyakov-et-al-iccv2019.pdf}
}
#################################################################


########################dataset#########################
Change the MVSEC/DSEC dataset address in dataset.py. 
################################################################


#########################code for mvsec############################

code_of_mvsec/DTC_pds_for_mvsec
usage: DTC-pds

code_of_mvsec/DTC_SPADE_for_mvsec
usage: DTC-spade

For training/test procedure, just execute:
bash train.sh/test.sh

####################################################################



###########################code for Dsec#############################

code_of_Dsec/LTC_Dsec
usage: generate DSEC website test result by DTC-PDS

code_of_Dsec/LTC_Dsec_spade
usage: generate DSEC website test result by DTC-SPADE

code_of_Dsec/LTC_for_Dsec/LTC_Dsec_clear_version
usage: DTC-PDS

code_of_Dsec/LTC_for_Dsec/LTC_Dsec_spade_clear_version
usage: DTC-SPADE

For training/test procedure, just execute:
bash train.sh/test.sh

#####################################################################


