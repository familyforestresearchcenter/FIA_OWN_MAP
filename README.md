# FIA_OWN_MAP

This repo contains code relevant to the processing and functions related to the FIA Ownership map generation. The files pertain to flowcharts, preprocessing, and analytical scripts to that end.

Updated: This script will be synced to push to the Umass OneDrive to be viewed and run there. All directory paths will be written to reflect the OneDrive file structure.

Overall work flow:
1. Run **Arc_Processing.ipynb**. This script will unpack the DMP ESRI geodatabase and do initial processing. *This only needs to be done when a new delivery of geodatabase files is available*.
2. Run **Random_Forest_Classifier_Traning_Script**. This script creates a generates a model able to classify ownership names, based on a training list derived from manual classification of DMP names for NWOS implementation. The training list contains confidential data and so must remain on an external flashdrive, but the derived model object is copied to the OUTPUTS/INTERMEDIATE folder. This script only needs to be run once, though, unless it becomes preferable to run it again on a new training set.
3. Run **Main.ipynb**. This script calls all the other scripts in the process, cycles through all of the states available in the INPUTS folder, and outputs the final spatial layers and tables.
