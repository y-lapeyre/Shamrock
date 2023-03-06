cd doxygen
doxygen dox.conf

cd ..
jupyter-book build guide_cpp
jupyter-book build guide_sycl
jupyter-book build shamrock_dev_doc
jupyter-book build shamrock_doc