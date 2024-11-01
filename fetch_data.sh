#/bin/bash


######################################### DISCLAIMER ###########################################
# If the <gdown> utlility for some reason doesn't work, here is the list of the                #
# direct links to the respective datasets in Google Drive. Download them manually              #
# and put them in the OUTPUT_DIR directory BEFORE executing the script.                        #
# Don't forget to set a USE_GDOWN flag to false/true! By default this flag is disabled.        #
#                                                                                              #
# * DDR Dataset:                                                                               #
#   - https://drive.google.com/drive/folders/1Ke7NKH5W_qHWgu2mHYkJZi6woVCvSg6e?usp=drive_link  #
#   - Download and put all 10 .zip parts in the OUTPUT_DIR.                                    #
#                                                                                              #
# * Messidor-2 Dataset:                                                                        #
#   - https://drive.google.com/drive/folders/1PDDMiKTFcZCHmhtgxwBqDFxl8gJaSQX_?usp=drive_link  #
#   - Download everything from this folder and put .csv and .zip files                         #
#     in the OUTPUT_DIR.                                                                       #
#                                                                                              #
# * APTOS Dataset:                                                                             #
#   - https://drive.google.com/file/d/1ERTxRtRs4hM65i-myBN5g1uBEEk7FLxf/view?usp=drive_link    #
#   - Download and put the single .zip file in the OUTPUT_DIR.                                 #
#                                                                                              #
# * FGADR Dataset:                                                                             #
#   - https://drive.google.com/file/d/1Y-0lCFmFVCBWtDNhT7fCi7-8pJ-0aiT2/view?usp=drive_link    #
#   - Download and put the single .zip file in the OUTPUT_DIR.                                 #
#                                                                                              #
# * IDRiD Dataset:                                                                             #
#   - https://drive.google.com/file/d/1v8nfmftnloovhwyP3EgnOSbvoDM3l9ud/view?usp=drive_link    #
#   -Download and put the single .zip file in the OUTPUT_DIR.                                  #
################################################################################################



OUTPUT_DIR="datasets"
USE_GDOWN=false
mkdir -p ${OUTPUT_DIR}


# DDR Dataset
# https://github.com/nkicsl/DDR-dataset
if [ ! -d "${OUTPUT_DIR}/DDR" ]; then
    if $USE_GDOWN; then
        # download parts from the Google Drive
        GDRIVE_LINK="1z6tSFmxW_aNayUqVxx6h6bY4kwGzUTEC"
        gdown --folder "${GDRIVE_LINK}" -O ${OUTPUT_DIR}
    fi

    # merge and unzip
    cat "${OUTPUT_DIR}"/DDR-dataset.zip* > "${OUTPUT_DIR}"/DDR-dataset.zip
    unzip "${OUTPUT_DIR}"/DDR-dataset.zip -d "${OUTPUT_DIR}"

    # cleanup
    mv "${OUTPUT_DIR}"/DDR-dataset/DR_grading "${OUTPUT_DIR}"/DDR
    rm -r "${OUTPUT_DIR}"/DDR-dataset*

    # fix labels format
    echo "filename,label" > ${OUTPUT_DIR}/DDR/train.csv
    awk '{printf "%s,%s\n", $1, $2}' ${OUTPUT_DIR}/DDR/train.txt >> ${OUTPUT_DIR}/DDR/train.csv
    rm ${OUTPUT_DIR}/DDR/train.txt

    echo "filename,label" > ${OUTPUT_DIR}/DDR/test.csv
    awk '{printf "%s,%s\n", $1, $2}' ${OUTPUT_DIR}/DDR/test.txt >> ${OUTPUT_DIR}/DDR/test.csv
    rm ${OUTPUT_DIR}/DDR/test.txt

    echo "filename,label" > ${OUTPUT_DIR}/DDR/val.csv
    awk '{printf "%s,%s\n", $1, $2}' ${OUTPUT_DIR}/DDR/valid.txt >> ${OUTPUT_DIR}/DDR/val.csv
    rm ${OUTPUT_DIR}/DDR/valid.txt
fi


# Messidor-2 Dataset
# https://www.adcis.net/en/third-party/messidor2/
if [ ! -d "${OUTPUT_DIR}/Messidor" ]; then
    if $USE_GDOWN; then
        # download parts from the Google Drive
        GDRIVE_LINK="1PDDMiKTFcZCHmhtgxwBqDFxl8gJaSQX_"
        gdown --folder "${GDRIVE_LINK}" -O "${OUTPUT_DIR}"
    fi

    # merge and unzip
    cat "${OUTPUT_DIR}"/IMAGES.zip* > "${OUTPUT_DIR}"/IMAGES.zip
    unzip "${OUTPUT_DIR}"/IMAGES.zip -d "${OUTPUT_DIR}"

    # cleanup
    mkdir -p "${OUTPUT_DIR}"/Messidor
    mv "${OUTPUT_DIR}"/IMAGES "${OUTPUT_DIR}"/Messidor/train
    mv "${OUTPUT_DIR}"/messidor_data.csv "${OUTPUT_DIR}"/Messidor/train_tmp.csv

    rm -r "${OUTPUT_DIR}"/IMAGES*
    rm "${OUTPUT_DIR}"/messidor_readme.txt "${OUTPUT_DIR}"/messidor-2.csv

    # fix labels format
    echo "filename,label" > ${OUTPUT_DIR}/Messidor/train.csv
    awk -F, 'NR>1 {printf "%s,%s\n", $1, $2}' ${OUTPUT_DIR}/Messidor/train_tmp.csv >> ${OUTPUT_DIR}/Messidor/train.csv
    rm ${OUTPUT_DIR}/Messidor/train_tmp.csv
fi


# APTOS Dataset
# https://www.kaggle.com/competitions/aptos2019-blindness-detection
if [ ! -d "${OUTPUT_DIR}/APTOS" ]; then
    if $USE_GDOWN; then
        # download from the Google Drive
        GDRIVE_LINK="1ERTxRtRs4hM65i-myBN5g1uBEEk7FLxf"
        gdown "${GDRIVE_LINK}" -O "${OUTPUT_DIR}"
    fi

    # unzip
    unzip "${OUTPUT_DIR}"/aptos2019-blindness-detection.zip -d "${OUTPUT_DIR}"/APTOS_tmp
    rm "${OUTPUT_DIR}"/aptos2019-blindness-detection.zip

    # cleanup
    mkdir -p "${OUTPUT_DIR}"/APTOS
    mv "${OUTPUT_DIR}"/APTOS_tmp/aptos2019-blindness-detection/train_images  "${OUTPUT_DIR}"/APTOS/train
    mv "${OUTPUT_DIR}"/APTOS_tmp/aptos2019-blindness-detection/test_images  "${OUTPUT_DIR}"/APTOS/test
    mv "${OUTPUT_DIR}"/APTOS_tmp/aptos2019-blindness-detection/train.csv  "${OUTPUT_DIR}"/APTOS/train_tmp.csv
    mv "${OUTPUT_DIR}"/APTOS_tmp/aptos2019-blindness-detection/test.csv  "${OUTPUT_DIR}"/APTOS/test_tmp.csv
    mv "${OUTPUT_DIR}"/APTOS_tmp/aptos2019-blindness-detection/sample_submission.csv  "${OUTPUT_DIR}"/APTOS/sample_submission.csv
    rm -r "${OUTPUT_DIR}"/APTOS_tmp

    # fix labels format
    echo "filename,label" > ${OUTPUT_DIR}/APTOS/train.csv
    awk -F, 'NR>1 {printf "%s.png,%s\n", $1, $2}' ${OUTPUT_DIR}/APTOS/train_tmp.csv >> ${OUTPUT_DIR}/APTOS/train.csv
    rm ${OUTPUT_DIR}/APTOS/train_tmp.csv

    echo "filename,label" > ${OUTPUT_DIR}/APTOS/test.csv
    awk -F, 'NR>1 {printf "%s.png,\n", $1}' ${OUTPUT_DIR}/APTOS/test_tmp.csv >> ${OUTPUT_DIR}/APTOS/test.csv
    rm ${OUTPUT_DIR}/APTOS/test_tmp.csv
fi


# FGADR Dataset
# https://paperswithcode.com/dataset/fgadr
if [ ! -d "${OUTPUT_DIR}/FGADR" ]; then
    if $USE_GDOWN; then
        # download from the Google Drive
        GDRIVE_LINK=1Y-0lCFmFVCBWtDNhT7fCi7-8pJ-0aiT2
        gdown "${GDRIVE_LINK}" -O "${OUTPUT_DIR}"
    fi

    # unzip
    unzip "${OUTPUT_DIR}"/FGADR-Seg-set_Release.zip -d "${OUTPUT_DIR}"/FGADR_tmp
    rm "${OUTPUT_DIR}"/FGADR-Seg-set_Release.zip

    # cleanup
    mkdir -p "${OUTPUT_DIR}"/FGADR
    mv "${OUTPUT_DIR}"/FGADR_tmp/Seg-set/Original_Images "${OUTPUT_DIR}"/FGADR/train
    mv "${OUTPUT_DIR}"/FGADR_tmp/Seg-set/DR_Seg_Grading_Label.csv "${OUTPUT_DIR}"/FGADR/train_tmp.csv
    rm -r "${OUTPUT_DIR}"/FGADR_tmp

    # fix labels format
    echo "filename,label" > ${OUTPUT_DIR}/FGADR/train.csv
    awk -F, '{printf "%s,%s\n", $1, $2}' ${OUTPUT_DIR}/FGADR/train_tmp.csv >> ${OUTPUT_DIR}/FGADR/train.csv
    rm ${OUTPUT_DIR}/FGADR/train_tmp.csv
fi

# IDRiD Dataset
# https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid
if [ ! -d "${OUTPUT_DIR}/IDRiD" ]; then
    if $USE_GDOWN; then
        # download from the Google Drive
        GDRIVE_LINK="1v8nfmftnloovhwyP3EgnOSbvoDM3l9ud"
        gdown "${GDRIVE_LINK}" -O "${OUTPUT_DIR}"
    fi

    # unzip
    unzip "${OUTPUT_DIR}/B. Disease Grading.zip" -d "${OUTPUT_DIR}"/IDRiD_tmp
    rm "${OUTPUT_DIR}/B. Disease Grading.zip"

    # cleanup
    mkdir -p "${OUTPUT_DIR}/IDRiD"
    mv "${OUTPUT_DIR}/IDRiD_tmp/B. Disease Grading/1. Original Images/a. Training Set" "${OUTPUT_DIR}"/IDRiD/train
    mv "${OUTPUT_DIR}/IDRiD_tmp/B. Disease Grading/1. Original Images/b. Testing Set" "${OUTPUT_DIR}"/IDRiD/test
    mv "${OUTPUT_DIR}/IDRiD_tmp/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv" "${OUTPUT_DIR}/IDRiD/train_tmp.csv"
    mv "${OUTPUT_DIR}/IDRiD_tmp/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv" "${OUTPUT_DIR}/IDRiD/test_tmp.csv"
    rm -r "${OUTPUT_DIR}"/IDRiD_tmp

    # fix labels format
    echo "filename,label" > ${OUTPUT_DIR}/IDRiD/train.csv
    awk -F, 'NR>1 {printf "%s.jpg,%s\n", $1, $2}' ${OUTPUT_DIR}/IDRiD/train_tmp.csv >> ${OUTPUT_DIR}/IDRiD/train.csv
    rm ${OUTPUT_DIR}/IDRiD/train_tmp.csv

    echo "filename,label" > ${OUTPUT_DIR}/IDRiD/test.csv
    awk -F, 'NR>1 {printf "%s.jpg,%s\n", $1, $2}' ${OUTPUT_DIR}/IDRiD/test_tmp.csv >> ${OUTPUT_DIR}/IDRiD/test.csv
    rm ${OUTPUT_DIR}/IDRiD/test_tmp.csv
fi
