DATA_ROOT = '/content/drive/MyDrive/fsdl_project'
DATA_DEST = '/content/land_cover_classification_unet/data'
WORKSPACE_ROOT = '/content/land_cover_classification_unet'
cp $DATA_ROOT/deepglobe2018-landcover-segmentation-traindataset.zip $DATA_DEST
unzip /content/land_cover_classification_unet/data/deepglobe2018-landcover-segmentation-traindataset.zip -d $DATA_DEST
unzip -qqq /content/land_cover_classification_unet/data/data.zip -d /content/land_cover_classification_unet/data
cd $WORKSPACE_ROOT
mv  -v data/data/* data/
rm -d data/data/