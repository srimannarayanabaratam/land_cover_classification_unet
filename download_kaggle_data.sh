
function usage() {
   echo "Usage: $0 -u kaggle_user -k kaggle_api_key [-d data_directory] "
   echo "  Download the deepglobe2018-landcover-segmentation-traindataset"
   echo "  Parameters: "
   echo "  -d data_directory      (optional) Local path to where the data should be stored (defaults to data)"
   echo "  Make also sure that the environment variables KAGGLE_USERNAME and KAGGLE_KEY are set using "
   echo "     export KAGGLE_USERNAME=xxx"
   echo "     export KAGGLE_KEY=xxx"
   exit 1
}

while getopts ":u:k:d:" arg; do
   case "${arg}" in
    d)
       data_directory="${OPTARG}"
       ;;
   esac
done


if [ -z "${KAGGLE_USERNAME}" ] || [ -z "${KAGGLE_KEY}" ]; then
   echo "Missing required options."
   usage
fi

if [ -z "${data_directory}" ]; then
   data_directory="data"
fi

kaggle datasets download -d geoap96/deepglobe2018-landcover-segmentation-traindataset
unzip deepglobe2018-landcover-segmentation-traindataset.zip -d ${data_directory}
rm deepglobe2018-landcover-segmentation-traindataset.zip
unzip -qqq ${data_directory}/data.zip -d ${data_directory}
rm ${data_directory}/data.zip
chmod 700 -R ${data_directory}/data
mv  -v ${data_directory}/data/* ${data_directory}/
rm -d ${data_directory}/data/