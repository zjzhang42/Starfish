#!/bin/bash
#
# download PHOENIX HiRes model spectra (enlightened by Ian Czekala)
# could work on the bigmac/westfield (bigmac/westfield => Dropbox => local)
# PHOENIX: ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/
#
#
# Instructions:
# >>> scp download_PHOENIX.sh bigmac/westfield:/path/ 
# >>> bigmac / westfield
# >>> screen
# ... run this bash script
# ... Ctrl+a+d <detach screen>
# hiking & beach & surfing
# >>> screen -r <re-attach to the screen>
# ... Ctrl+a+d <detach screen>
# #--
#
##########################################
## Author: ZJ Zhang (zhoujian@hawaii.edu)
## Jul. 06th, 2017
##########################################

####### MODIFICATION HISTORY:
#
# ZJ Zhang (Jul. 06th, 2017)
#
#############################

### this is what you need to revise every time when you use this bash script (DIRECTORY is a relative path!)
DIRECTORY="Spec_Lib/Spec_for_Starfish/libraries/raw/PHOENIX/"


### 1. Setting Download directory
# Check to see if the directory exists, if not, make one -
if [ ! -d "$DIRECTORY" ]; then
    echo $DIRECTORY does not exist, creating.
        mkdir -p $DIRECTORY
fi
# --

### 2. Start downloading...
# 2.1 Preparation
cd $DIRECTORY
# 2.2 Model Spectra
wget -r --no-parent ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/
# --

### 3. Move PHOENIX folder to the target workplace
# could do it by hand
# --

#### END
