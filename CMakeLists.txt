set(LPDNN_MODEL_DATASET "imagenet")
set(LPDNN_MODEL_DATASET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../samples/ZF)
set(LPDNN_MODEL_SPEC ${CMAKE_CURRENT_SOURCE_DIR}/proto/deploy.prototxt)
set(LPDNN_MODEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/model/float32/model.caffemodel)

set(LPDNN_IMAGE_SIZE "227" "227")
set(LPDNN_IMAGE_COLOR "BGR")
set(LPDNN_IMAGE_MEAN "127.5" "127.5" "127.5")
set(LPDNN_IMAGE_SCALE "0.007843")
set(LPDNN_IMAGE_MEAN_TYPE "pixel")

include(${CMAKE_SOURCE_DIR}/app/classification/CMakeLists.txt)

