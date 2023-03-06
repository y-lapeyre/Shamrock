cd doxygen
doxygen dox.conf

cd ..
jupyter-book build guide_cpp
jupyter-book build guide_sycl
jupyter-book build shamrock_dev_doc
jupyter-book build shamrock_doc

rm -rf _build
mkdir _build
cd _build


mkdir guide_cpp
mkdir guide_sycl
mkdir shamrock_dev_doc
mkdir shamrock_doc

mkdir doxygen

cp -r ../guide_cpp/_build/html/* guide_cpp
cp -r ../guide_sycl/_build/html/* guide_sycl
cp -r ../shamrock_dev_doc/_build/html/* shamrock_dev_doc
cp -r ../shamrock_doc/_build/html/* shamrock_doc


cp -r ../doxygen/html/* doxygen


cp ../tmpindex.html index.html